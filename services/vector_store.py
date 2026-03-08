"""
services/vector_store.py
-------------------------
Two-path RAG architecture:

  PATH A — Account queries (balance, SIP, transactions):
    PAN/ID exact lookup → JSON → instant (~0ms)
    Used for: portfolio_enquiry, account_statement, transaction_status, redemption_request

  PATH B — Knowledge queries (KYC rules, fund info, SEBI norms):
    Groq embed query → ChromaDB cosine search → top-k docs → LLM context (~300ms)
    Used for: compliance_query, general_enquiry, dividend_info, kyc_update

This way:
  - Account queries stay under 3s total ✅
  - Knowledge queries get accurate grounded answers ✅
  - No sentence-transformers, no local model loading ✅
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import httpx

logger    = logging.getLogger(__name__)
DATA_PATH = Path(__file__).parent.parent / "data" / "investors.json"
KB_PATH   = Path(__file__).parent.parent / "data" / "knowledge_base.json"
DB_PATH   = Path(__file__).parent.parent / "data" / "chroma_db"

try:
    import chromadb
    CHROMA_OK = True
except ImportError:
    CHROMA_OK = False
    logger.warning("chromadb not installed — knowledge RAG disabled. pip install chromadb")

# ── Singleton ─────────────────────────────────────────────────────────────────
_store: Optional["VectorStore"] = None

def get_vector_store() -> "VectorStore":
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


class VectorStore:
    """
    Dual-path retrieval:
      - Investor records  → exact JSON lookup (instant)
      - Knowledge base    → Groq embedding + ChromaDB (for policy/FAQ queries)
    """

    def __init__(self) -> None:
        from config import settings
        self._settings   = settings
        self._investors: list[dict] = []
        self._kb_docs:   list[dict] = []
        self._collection = None
        self._chroma_ok  = False

        self._load_investors()
        self._load_knowledge_base()
        if CHROMA_OK:
            self._init_chroma()

    # ── Loaders ───────────────────────────────────────────────────────────────
    def _load_investors(self) -> None:
        if not DATA_PATH.exists():
            logger.warning("investors.json not found. Run: python setup_data.py")
            return
        with open(DATA_PATH) as f:
            self._investors = json.load(f)
        logger.info(f"Loaded {len(self._investors)} investor records (exact lookup)")

    def _load_knowledge_base(self) -> None:
        if not KB_PATH.exists():
            logger.info("No knowledge_base.json — knowledge RAG disabled")
            return
        with open(KB_PATH) as f:
            self._kb_docs = json.load(f)
        logger.info(f"Loaded {len(self._kb_docs)} knowledge base documents")

    # ── Groq Embeddings ───────────────────────────────────────────────────────
    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        """Sync embed via Groq nomic-embed-text-v1_5. Used during indexing."""
        resp = httpx.post(
            "https://api.groq.com/openai/v1/embeddings",
            headers = {"Authorization": f"Bearer {self._settings.groq_api_key}",
                       "Content-Type": "application/json"},
            json    = {"model": "nomic-embed-text-v1_5", "input": texts},
            timeout = 30.0,
        )
        resp.raise_for_status()
        return [d["embedding"] for d in resp.json()["data"]]

    async def _embed_async(self, texts: list[str]) -> list[list[float]]:
        """Async embed via Groq — used at query time to avoid blocking."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._embed_sync(texts))

    # ── ChromaDB — Knowledge Base only ───────────────────────────────────────
    def _init_chroma(self) -> None:
        if not self._kb_docs:
            logger.info("No KB docs to index — skipping ChromaDB init")
            return
        try:
            DB_PATH.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(DB_PATH))
            self._collection = client.get_or_create_collection(
                name     = "knowledge_base",
                metadata = {"hnsw:space": "cosine"},
            )
            if self._collection.count() == 0:
                logger.info("Indexing knowledge base via Groq embeddings...")
                self._index_kb()
            self._chroma_ok = True
            logger.info(f"ChromaDB ready | {self._collection.count()} KB documents indexed")
        except Exception as e:
            logger.warning(f"ChromaDB init failed: {e}")
            self._chroma_ok = False

    def _index_kb(self) -> None:
        ids, texts, metas = [], [], []
        for i, doc in enumerate(self._kb_docs):
            ids.append(f"kb_{i}")
            texts.append(doc["content"])
            metas.append({"title": doc.get("title",""), "category": doc.get("category","")})
        embeddings = self._embed_sync(texts)
        self._collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metas)
        logger.info(f"Indexed {len(ids)} KB documents")

    # ═════════════════════════════════════════════════════════════════════════
    # PATH A — Account data (exact lookup, instant)
    # ═════════════════════════════════════════════════════════════════════════
    def get_investor(
        self,
        pan: Optional[str]         = None,
        investor_id: Optional[str] = None,
        query: str                 = "",
    ) -> Optional[dict]:
        """
        Exact lookup only — never guesses.
        Returns investor record or None (caller must ask for PAN).
        """
        if not self._investors:
            return None

        # Exact PAN
        if pan:
            for inv in self._investors:
                if inv["pan"] == pan.upper().strip():
                    logger.info(f"Investor lookup: PAN match → {inv['name']}")
                    return inv

        # Exact investor_id
        if investor_id:
            for inv in self._investors:
                if inv["investor_id"] == investor_id.upper().strip():
                    logger.info(f"Investor lookup: ID match → {inv['name']}")
                    return inv

        # PAN or ID explicitly in query text
        if query:
            q = query.upper()
            for inv in self._investors:
                if inv["pan"] in q or inv["investor_id"] in q:
                    logger.info(f"Investor lookup: keyword match → {inv['name']}")
                    return inv

        return None

    # ═════════════════════════════════════════════════════════════════════════
    # PATH B — Knowledge RAG (Groq embed + ChromaDB, ~300ms)
    # ═════════════════════════════════════════════════════════════════════════
    async def search_knowledge(self, query: str, top_k: int = 2) -> str:
        """
        Semantic search over knowledge base (KYC rules, SEBI norms, fund FAQs).
        Returns formatted context string for LLM injection.
        Returns empty string if ChromaDB unavailable or no good match.
        """
        if not self._chroma_ok or not self._collection or not query:
            return ""

        try:
            q_embedding = (await self._embed_async([query]))[0]
            results     = self._collection.query(
                query_embeddings = [q_embedding],
                n_results        = top_k,
            )
            if not results or not results["ids"] or not results["ids"][0]:
                return ""

            chunks = []
            for i, doc_text in enumerate(results["documents"][0]):
                dist  = results["distances"][0][i]
                meta  = results["metadatas"][0][i]
                if dist < 0.5:   # Only include confident matches
                    chunks.append(f"[{meta.get('title','KB')}]\n{doc_text}")

            if chunks:
                context = "\n\n".join(chunks)
                logger.info(f"KB RAG: {len(chunks)} chunks retrieved for query '{query[:40]}'")
                return context

        except Exception as e:
            logger.warning(f"KB RAG search failed: {e}")

        return ""

    def get_all_investors(self) -> list[dict]:
        return self._investors

    def reindex_kb(self) -> None:
        """Force re-index knowledge base (use after updating knowledge_base.json)."""
        if self._collection:
            try:
                self._collection.delete(where={"category": {"$ne": ""}})
            except Exception:
                pass
        self._index_kb()
"""
setup_data.py — Run once before starting the app.

    python setup_data.py

Steps:
  1. Generate 20 synthetic investor records → data/investors.json
  2. Generate 10 CAMS knowledge base docs  → data/knowledge_base.json
  3. Embed KB docs via Groq + index into ChromaDB → data/chroma_db/

After this, investor lookup is instant (JSON).
Knowledge queries use ChromaDB RAG via Groq embeddings.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("  CAMS Assistant — Data Setup (Two-Path RAG)")
print("=" * 60)

# ── Step 1: Investor data ─────────────────────────────────────────────────────
print("\n[1/3] Generating synthetic investor records...")
from data.synthetic_investors import generate_investors
investors = generate_investors(20)
out = Path("data/investors.json")
out.parent.mkdir(exist_ok=True)
with open(out, "w") as f:
    json.dump(investors, f, indent=2)
print(f"  ✅ {len(investors)} investor records → {out}")
print(f"  (PATH A: exact PAN/ID lookup — instant, no embedding needed)")

# ── Step 2: Knowledge base ────────────────────────────────────────────────────
print("\n[2/3] Generating CAMS knowledge base documents...")
from data.knowledge_base_generator import KNOWLEDGE_BASE
kb_out = Path("data/knowledge_base.json")
with open(kb_out, "w") as f:
    json.dump(KNOWLEDGE_BASE, f, indent=2)
print(f"  ✅ {len(KNOWLEDGE_BASE)} KB documents → {kb_out}")
categories = set(d["category"] for d in KNOWLEDGE_BASE)
print(f"  Categories: {', '.join(sorted(categories))}")

# ── Step 3: Embed KB + index into ChromaDB ────────────────────────────────────
print("\n[3/3] Embedding KB documents via Groq + indexing into ChromaDB...")
print("  (PATH B: Groq nomic-embed-text-v1_5 → ChromaDB cosine search)")
try:
    from services.vector_store import VectorStore
    vs = VectorStore()
    if vs._chroma_ok:
        print(f"  ✅ ChromaDB ready | {vs._collection.count()} KB documents indexed")
        print(f"  Embedding model: nomic-embed-text-v1_5 (Groq API, no local model)")
    elif not vs._kb_docs:
        print("  ⚠️  No KB docs loaded")
    else:
        print("  ⚠️  ChromaDB unavailable — PATH B disabled, PATH A still works")
        print("     pip install chromadb")
except Exception as e:
    print(f"  ❌ Failed: {e}")
    print("     Make sure GROQ_API_KEY is set in .env")
    sys.exit(1)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Architecture:")
print("  PATH A — Account queries (balance, SIP, transactions)")
print("           PAN/ID exact lookup → instant → ~3s total")
print("  PATH B — Knowledge queries (KYC, SEBI, fund info)")
print("           Groq embed → ChromaDB → ~300ms → ~3.5s total")
print("\n  Test PANs:")
for inv in investors[:5]:
    print(f"    {inv['pan']}  {inv['name']:20s}  ₹{inv['total_current_value']:>12,.0f}")
print("\n  Run:")
print("    python main.py       # FastAPI on :8000")
print("    streamlit run app.py # UI on :8501")
print("=" * 60)
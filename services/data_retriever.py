"""
services/data_retriever.py
---------------------------
Smart context retriever — decides which RAG path to use based on intent.

PATH A (Account data — instant):
  portfolio_enquiry, account_statement, transaction_status,
  redemption_request, dividend_info
  → Exact PAN/ID lookup from investors.json

PATH B (Knowledge — ~300ms):
  compliance_query, kyc_update, general_enquiry
  → Groq embed + ChromaDB search over knowledge_base.json

Both paths return a formatted context string injected into the LLM prompt.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Intents that need account data (PATH A)
ACCOUNT_INTENTS = {
    "portfolio_enquiry",
    "account_statement",
    "transaction_status",
    "redemption_request",
    "dividend_info",
}

# Intents that need knowledge base (PATH B)
KNOWLEDGE_INTENTS = {
    "compliance_query",
    "kyc_update",
    "general_enquiry",
    "unknown",
}

PAN_RE   = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", re.IGNORECASE)
CAMSID_RE = re.compile(r"\bCAMS\d{5}\b", re.IGNORECASE)


def extract_identifiers(text: str) -> dict:
    pan_m = PAN_RE.search(text.upper())
    id_m  = CAMSID_RE.search(text.upper())
    return {
        "pan":         pan_m.group(0) if pan_m else None,
        "investor_id": id_m.group(0)  if id_m  else None,
    }


def format_investor_context(investor: dict) -> str:
    """Format investor record as LLM-readable context string."""
    lines = [
        "INVESTOR ACCOUNT DATA:",
        f"  Name:           {investor['name']}",
        f"  PAN:            {investor['pan']}",
        f"  Investor ID:    {investor['investor_id']}",
        f"  KYC Status:     {investor['kyc_status']}",
        f"  City:           {investor['city']}",
        f"  Member Since:   {investor['account_since']}",
        "",
        "PORTFOLIO SUMMARY:",
        f"  Total Invested:  ₹{investor['total_invested']:,.2f}",
        f"  Current Value:   ₹{investor['total_current_value']:,.2f}",
        f"  Total Gain/Loss: ₹{investor['total_gain_loss']:,.2f} ({investor['total_gain_loss_pct']:+.2f}%)",
        f"  Monthly SIP:     ₹{investor['total_sip_amount']:,}",
        f"  Number of Funds: {investor['num_funds']}",
        "",
        "FUND HOLDINGS:",
    ]
    for i, h in enumerate(investor["holdings"], 1):
        sign = "+" if h["gain_loss"] >= 0 else ""
        lines += [
            f"  {i}. {h['fund_name']} ({h['category']})",
            f"     Units: {h['units']} | NAV: ₹{h['nav']} | Value: ₹{h['current_value']:,.2f}",
            f"     Invested: ₹{h['invested_amount']:,.2f} | "
            f"Gain/Loss: {sign}₹{h['gain_loss']:,.2f} ({sign}{h['gain_loss_pct']}%)",
            f"     SIP: {'₹'+str(h['sip_amount'])+'/month on '+str(h['sip_date'])+'th' if h['sip_active'] else 'No active SIP'} | Option: {h['dividend_option']}",
        ]
    if investor.get("recent_transactions"):
        lines += ["", "RECENT TRANSACTIONS:"]
        for t in investor["recent_transactions"][:3]:
            lines.append(
                f"  [{t['date']}] {t['type']} | {t['fund_name']} | "
                f"₹{t['amount']:,} | {t['units']} units | {t['status']} | Ref: {t['txn_id']}"
            )
    return "\n".join(lines)


async def get_context(
    user_message: str,
    intent: Optional[str]      = None,
    pan: Optional[str]         = None,
    investor_id: Optional[str] = None,
) -> tuple[Optional[dict], str]:
    """
    Main retriever entry point.

    Args:
        user_message: Raw user text
        intent: Detected intent (used to choose path A or B)
        pan: PAN if already known from session
        investor_id: Investor ID if already known

    Returns:
        (investor_record_or_None, context_string_for_llm)
    """
    from services.vector_store import get_vector_store
    vs = get_vector_store()

    # Extract identifiers from message
    ids = extract_identifiers(user_message)
    pan         = pan         or ids["pan"]
    investor_id = investor_id or ids["investor_id"]

    # ── PATH A: Account data (exact lookup) ───────────────────────────────
    if intent in ACCOUNT_INTENTS or intent is None:
        investor = vs.get_investor(pan=pan, investor_id=investor_id, query=user_message)
        if investor:
            context = format_investor_context(investor)
            logger.info(f"PATH A: account data retrieved for {investor['name']}")
            return investor, context
        # No identifier found — return nothing, LLM will ask for PAN
        logger.info("PATH A: no identifier found — LLM will ask for PAN")
        return None, ""

    # ── PATH B: Knowledge base RAG ────────────────────────────────────────
    if intent in KNOWLEDGE_INTENTS:
        kb_context = await vs.search_knowledge(user_message, top_k=2)
        if kb_context:
            logger.info(f"PATH B: knowledge context retrieved ({len(kb_context)} chars)")
            return None, kb_context
        logger.info("PATH B: no KB match found")
        return None, ""

    return None, ""


async def get_investor_context(
    user_message: str,
    extracted_entities: Optional[dict] = None,
) -> tuple[Optional[dict], str]:
    """
    Backward-compatible wrapper used by main.py.
    Passes through to get_context without intent routing
    (intent not known yet at this stage — LLM determines it).
    """
    pan         = None
    investor_id = None
    if extracted_entities:
        investor_id = extracted_entities.get("investor_id")

    ids = extract_identifiers(user_message)
    pan         = ids["pan"]
    investor_id = investor_id or ids["investor_id"]

    from services.vector_store import get_vector_store
    vs       = get_vector_store()
    investor = vs.get_investor(pan=pan, investor_id=investor_id, query=user_message)

    if investor:
        context = format_investor_context(investor)
        logger.info(f"Investor context: {investor['name']} | ₹{investor['total_current_value']:,.0f}")
        return investor, context

    return None, ""
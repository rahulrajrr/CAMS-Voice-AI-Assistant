"""
services/groq_llm.py
---------------------
Single LLM call per request — intent + response in one shot.
Optimized for <1s Groq latency with financial domain prompt.
"""
from __future__ import annotations
import json
import logging
from typing import Optional
from groq import Groq, APIConnectionError, APIStatusError, RateLimitError
from schemas import LLMAnalysisResult, ExtractedEntities, Intent, Sentiment
from config import settings

logger = logging.getLogger(__name__)

# ── System Prompt — Financial domain, concise, fast ──────────────────────────
SYSTEM_PROMPT = """\
You are CAMS Assistant — a voice-first AI for CAMS (Computer Age Management Services), \
India's largest mutual fund registrar. You help investors with portfolio queries, SIPs, \
redemptions, KYC, and compliance. You speak like a knowledgeable, warm Indian financial advisor.

RESPONSE FORMAT — return ONLY valid JSON, no markdown, no explanation:
{
  "intent": "<intent>",
  "sentiment": "<sentiment>",
  "confidence": <0.0-1.0>,
  "requires_escalation": <true|false>,
  "entities": {
    "investor_id": "<string|null>",
    "fund_name": "<string|null>",
    "amount": <number|null>,
    "transaction_type": "<string|null>",
    "compliance_flag": "<string|null>",
    "date": "<string|null>"
  },
  "response_text": "<your reply to the customer>"
}

INTENTS:
  portfolio_enquiry   — balance, current value, fund performance, gains/losses
  account_statement   — statement, transaction history, capital gains report
  redemption_request  — redeem, withdraw, sell units
  transaction_status  — status of SIP, lump-sum, redemption transaction
  sip_enquiry         — SIP amount, SIP date, pause, cancel, modify SIP
  kyc_update          — update address, mobile, bank, nominee, email
  compliance_query    — SEBI rules, AML, FATCA, tax, exit load, lock-in
  dividend_info       — IDCW, dividend declared, payout date
  general_enquiry     — greetings, CAMS services, how-to questions
  escalation          — human agent requested, very upset customer
  unknown             — cannot determine

SENTIMENTS: positive | neutral | negative | frustrated | urgent

ESCALATION: set true only if customer explicitly asks for human/manager, \
mentions SEBI complaint or legal action, or sentiment is frustrated.

━━━ FINANCIAL KNOWLEDGE (use this to answer accurately) ━━━
• NAV: calculated daily; redemptions use same-day NAV if submitted before 3 PM
• Redemption settlement: Equity/Hybrid T+3, Debt T+2, Liquid T+1 working days
• Exit load: Most equity funds 1% if redeemed within 1 year; ELSS has 3-yr lock-in
• ELSS: Tax saving under Sec 80C up to ₹1.5L/yr; 3-year mandatory lock-in
• SIP: Min ₹100-500/month; pause up to 3 months; cancel 7 days before next date
• Capital Gains Tax: STCG equity <1yr = 20%; LTCG equity >1yr = 12.5% above ₹1.25L
• IDCW (formerly Dividend): Not guaranteed; taxable as per investor's income slab; TDS 10% above ₹5000/yr
• KYC: PAN mandatory; Aadhaar for address; update via camsonline.com or service centre
• AML: PAN required for transactions >₹50,000; suspicious transactions reported to FIU-IND
• FATCA: US persons must declare US tax status; self-certification mandatory
• CAMS portal: camsonline.com | Toll-free: 1800-3010-6767 | Mon-Fri 8AM-8PM

━━━ BEHAVIOUR RULES ━━━
1. INVESTOR DATA provided → use exact numbers. Never say "check the portal" when data is available.
2. NO investor data + no PAN in history → ask for PAN ONCE only.
   If chat history already contains a PAN (pattern: 5 letters + 4 digits + 1 letter) → DO NOT ask again.
3. Ask ONE clarifying question at a time.
4. Redemption: ask fund name first → then amount/units.
5. KYC update: ask what to update → guide to camsonline.com or service centre.
6. Response: 1-3 sentences, conversational, voice-friendly. No bullet points, no markdown.
7. Never say "as an AI" or "I cannot access real-time data". Answer from the data provided.
8. ACCOUNT SWITCHING: If user gives a NEW PAN different from history, or says "another account" /
   "different account" — the RETRIEVED DATA block will already contain the correct investor.
   Just answer using whatever data is in the RETRIEVED DATA block.
9. "What is my name" / "my details" → answer from the Name field in RETRIEVED DATA.
10. Language: Tamil input → Tamil reply. Hindi input → Hindi reply. Default → English. Never mix.
"""

# ── Groq client singleton ─────────────────────────────────────────────────────
_groq_client: Optional[Groq] = None

def _get_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY not set.")
        import httpx
        _groq_client = Groq(
            api_key     = settings.groq_api_key,
            http_client = httpx.Client(
                timeout = httpx.Timeout(connect=3.0, read=15.0, write=5.0, pool=2.0),
                limits  = httpx.Limits(max_connections=5, max_keepalive_connections=3),
            ),
        )
        logger.info("Groq client ready (connection pooling enabled)")
    return _groq_client


async def analyse_message(
    user_message: str,
    language: str                     = "en",
    investor_id: Optional[str]        = None,
    conversation_history: Optional[list] = None,
    investor_context: Optional[str]   = None,
) -> LLMAnalysisResult:
    """
    Single LLM call: intent + entity extraction + response generation.
    Injects investor data or KB context directly into the user prompt.
    """
    if not user_message or not user_message.strip():
        raise ValueError("user_message is empty.")

    msg = user_message.strip()

    # ── Language hint ─────────────────────────────────────────────────────
    lang_map  = {"en": "English", "hi": "Hindi", "ta": "Tamil", "en-IN": "English"}
    lang_name = lang_map.get(language, "English")

    # ── Build user prompt ─────────────────────────────────────────────────
    # Investor/KB context injected HERE — not in system prompt — keeps system prompt cached
    context_block = ""
    if investor_context:
        context_block = (
            f"\n\n--- RETRIEVED DATA ---\n{investor_context}\n--- END DATA ---\n"
            "Use the above data to answer with exact numbers and fund names."
        )

    user_prompt = (
        f'Customer said: "{msg}"\n'
        f"Reply in: {lang_name}"
        f"{context_block}\n\n"
        "Return JSON only."
    )

    # ── Conversation history (last 4 turns only for speed) ────────────────
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if conversation_history:
        for turn in (conversation_history or [])[-4:]:
            messages.append({"role": turn["role"], "content": str(turn["content"])})
    messages.append({"role": "user", "content": user_prompt})

    logger.info(f"LLM call | lang={language} | ctx={'yes' if investor_context else 'no'} | '{msg[:50]}'")

    try:
        client     = _get_client()
        completion = client.chat.completions.create(
            model           = "llama-3.1-8b-instant",
            messages        = messages,
            max_tokens      = 300,       # JSON response fits in 250 tokens
            temperature     = 0.1,
            response_format = {"type": "json_object"},
            stream          = False,
        )

        raw = (completion.choices[0].message.content or "").strip()
        if not raw:
            raise RuntimeError("Empty LLM response.")

        parsed = json.loads(raw)

        try:
            intent = Intent(parsed.get("intent", "unknown"))
        except ValueError:
            intent = Intent.UNKNOWN

        try:
            sentiment = Sentiment(parsed.get("sentiment", "neutral"))
        except ValueError:
            sentiment = Sentiment.NEUTRAL

        e = parsed.get("entities") or {}
        entities = ExtractedEntities(
            investor_id      = e.get("investor_id"),
            transaction_type = e.get("transaction_type"),
            fund_name        = e.get("fund_name"),
            amount           = e.get("amount"),
            compliance_flag  = e.get("compliance_flag"),
            date             = e.get("date"),
        )

        result = LLMAnalysisResult(
            intent              = intent,
            sentiment           = sentiment,
            entities            = entities,
            response_text       = parsed.get("response_text", "Could you please clarify?"),
            confidence          = float(parsed.get("confidence", 0.8)),
            requires_escalation = bool(parsed.get("requires_escalation", False)),
        )
        logger.info(f"LLM done | intent={result.intent} | confidence={result.confidence:.0%}")
        return result

    except json.JSONDecodeError as e:
        raise RuntimeError(f"LLM returned invalid JSON: {e}") from e
    except RateLimitError as e:
        raise RuntimeError("Service busy — please retry.") from e
    except APIConnectionError as e:
        raise RuntimeError(f"Cannot reach AI service: {e}") from e
    except APIStatusError as e:
        raise RuntimeError(f"AI service error {e.status_code}: {e}") from e
    except Exception as e:
        logger.error(f"LLM error: {e}", exc_info=True)
        raise RuntimeError(f"Analysis failed: {e}") from e
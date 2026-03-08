"""
action_engine.py
-----------------
Action Engine for the Voice & Chat Assistant.
Evaluates the LLM analysis result and triggers appropriate backend actions
based on the detected intent — with mandatory compliance checks before
executing any financial transactions.

Compliance-first design:
  - Every action passes through a compliance gate
  - Compliance flags (AML, SEBI, FATCA) block action execution
  - All actions and compliance decisions are audit-logged
  - No financial action is executed without a clean compliance check
"""

import uuid
import logging
from schemas import (
    LLMAnalysisResult,
    ActionResult,
    ActionStatus,
    Intent,
    Sentiment,
)

# ── Module Logger ────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ── Compliance Keywords ───────────────────────────────────────────────────────
# Any of these flags in extracted entities will block action execution
# and trigger a compliance escalation for human review.
COMPLIANCE_BLOCK_KEYWORDS = {
    "aml", "fatca", "suspicious", "fraud", "money laundering",
    "sebi complaint", "court", "legal", "freeze", "blocked",
}


# ── Compliance Check ──────────────────────────────────────────────────────────
def _run_compliance_check(analysis: LLMAnalysisResult) -> tuple[bool, str]:
    """
    Run a compliance gate check before executing any financial action.

    Checks for:
      1. Compliance flags in extracted entities (AML, FATCA, SEBI keywords)
      2. Suspicious transaction amounts (> ₹10,00,000 triggers AML review)
      3. Escalation flag set by the LLM

    Args:
        analysis (LLMAnalysisResult): The LLM analysis output containing
                                      entities and escalation flags.

    Returns:
        tuple[bool, str]:
            - bool: True if compliance passed (safe to proceed), False if blocked.
            - str: Human-readable reason for the compliance decision.
    """
    logger.info("Running compliance check...")

    # ── Check 1: Compliance flag in entities ─────────────────────────────
    compliance_flag = analysis.entities.compliance_flag
    if compliance_flag:
        flag_lower = compliance_flag.lower()
        for keyword in COMPLIANCE_BLOCK_KEYWORDS:
            if keyword in flag_lower:
                reason = (
                    f"Compliance flag detected: '{compliance_flag}'. "
                    f"Action blocked pending compliance team review."
                )
                logger.warning(f"Compliance BLOCKED | reason={reason}")
                return False, reason

    # ── Check 2: High-value transaction AML threshold ────────────────────
    amount = analysis.entities.amount
    if amount and amount > 1_000_000:  # ₹10 Lakh threshold
        reason = (
            f"High-value transaction of ₹{amount:,.2f} detected. "
            f"AML review required before processing."
        )
        logger.warning(f"Compliance BLOCKED (AML threshold) | amount={amount} | reason={reason}")
        return False, reason

    # ── Check 3: Escalation flag ──────────────────────────────────────────
    if analysis.requires_escalation and analysis.intent == Intent.ESCALATION:
        reason = "Customer explicitly requested escalation. Routing to human agent."
        logger.info(f"Compliance CHECK — escalation triggered | reason={reason}")
        return False, reason

    logger.info("Compliance check PASSED. Action may proceed.")
    return True, "Compliance check passed."


# ── Action Dispatcher ─────────────────────────────────────────────────────────
async def trigger_action(analysis: LLMAnalysisResult) -> ActionResult:
    """
    Evaluate the LLM analysis and trigger the appropriate backend action.

    Flow:
      1. Run compliance check — block if flagged.
      2. Dispatch to the appropriate intent handler.
      3. Return a structured ActionResult with status, reference ID, and message.

    Args:
        analysis (LLMAnalysisResult): The full NLP analysis result from Groq LLM,
                                      containing intent, entities, and escalation flag.

    Returns:
        ActionResult: The outcome of the triggered (or blocked) action, including:
                      - action_taken: What was attempted.
                      - status: Success / Blocked / Pending / Not Required.
                      - reference_id: Unique ID for audit trail.
                      - message: Human-readable outcome description.
                      - compliance_checked: Whether compliance gate was run.

    Raises:
        RuntimeError: If an unexpected error occurs during action dispatch.
    """
    logger.info(
        f"Action engine triggered | intent={analysis.intent} | "
        f"sentiment={analysis.sentiment} | escalation={analysis.requires_escalation}"
    )

    reference_id = str(uuid.uuid4())[:12].upper()  # Short reference ID for audit

    try:
        # ── Step 1: Compliance Gate ───────────────────────────────────────
        compliance_passed, compliance_reason = _run_compliance_check(analysis)

        if not compliance_passed:
            logger.warning(
                f"Action BLOCKED by compliance gate | ref={reference_id} | "
                f"reason={compliance_reason}"
            )
            return ActionResult(
                action_taken      = f"Attempted: {analysis.intent.value}",
                status            = ActionStatus.BLOCKED_COMPLIANCE,
                reference_id      = reference_id,
                message           = compliance_reason,
                compliance_checked = True,
            )

        # ── Step 2: Intent-based Action Dispatch ─────────────────────────
        intent = analysis.intent

        if intent == Intent.REDEMPTION_REQUEST:
            return await _handle_redemption(analysis, reference_id)

        elif intent == Intent.ACCOUNT_STATEMENT:
            return await _handle_account_statement(analysis, reference_id)

        elif intent == Intent.COMPLIANCE_QUERY:
            return await _handle_compliance_query(analysis, reference_id)

        elif intent == Intent.PORTFOLIO_ENQUIRY:
            return await _handle_portfolio_enquiry(analysis, reference_id)

        elif intent == Intent.TRANSACTION_STATUS:
            return await _handle_transaction_status(analysis, reference_id)

        elif intent == Intent.KYC_UPDATE:
            return await _handle_kyc_update(analysis, reference_id)

        elif intent in (Intent.ESCALATION, Intent.UNKNOWN):
            return await _handle_escalation(analysis, reference_id)

        else:
            # General enquiry or dividend info — informational, no action needed
            logger.info(f"Intent '{intent}' requires no backend action.")
            return ActionResult(
                action_taken      = f"Informational response for: {intent.value}",
                status            = ActionStatus.NOT_REQUIRED,
                reference_id      = reference_id,
                message           = "No backend action required for this query.",
                compliance_checked = True,
            )

    except Exception as e:
        logger.error(
            f"Unexpected error in action engine | intent={analysis.intent} | "
            f"ref={reference_id} | error={str(e)}",
            exc_info=True,
        )
        raise RuntimeError(
            f"Action engine failed for intent '{analysis.intent}': {str(e)}"
        ) from e


# ── Intent Handlers ───────────────────────────────────────────────────────────

async def _handle_redemption(analysis: LLMAnalysisResult, ref_id: str) -> ActionResult:
    """
    Handle a mutual fund redemption request.

    In production, this would call the CAMS redemption API.
    Currently simulates successful initiation for prototype purposes.

    Args:
        analysis (LLMAnalysisResult): NLP result with entity details.
        ref_id (str): Audit reference ID.

    Returns:
        ActionResult: Redemption initiation result.
    """
    logger.info(
        f"Processing redemption request | investor={analysis.entities.investor_id} | "
        f"fund={analysis.entities.fund_name} | amount={analysis.entities.amount} | "
        f"ref={ref_id}"
    )

    # TODO: Replace with actual CAMS Redemption API call
    # e.g., await cams_api.initiate_redemption(investor_id, fund_name, amount)

    return ActionResult(
        action_taken       = "Redemption request initiated",
        status             = ActionStatus.PENDING,
        reference_id       = ref_id,
        message            = (
            f"Your redemption request has been received and is being processed. "
            f"Reference ID: {ref_id}. You will receive a confirmation within 1–2 business days."
        ),
        compliance_checked = True,
    )


async def _handle_account_statement(analysis: LLMAnalysisResult, ref_id: str) -> ActionResult:
    """
    Handle an account statement request.

    Args:
        analysis (LLMAnalysisResult): NLP result with investor details.
        ref_id (str): Audit reference ID.

    Returns:
        ActionResult: Statement request result.
    """
    logger.info(
        f"Processing account statement request | investor={analysis.entities.investor_id} | "
        f"ref={ref_id}"
    )

    # TODO: Replace with actual CAMS Statement API call

    return ActionResult(
        action_taken       = "Account statement request processed",
        status             = ActionStatus.SUCCESS,
        reference_id       = ref_id,
        message            = (
            f"Your account statement has been queued for generation. "
            f"It will be sent to your registered email within 24 hours. Ref: {ref_id}."
        ),
        compliance_checked = True,
    )


async def _handle_compliance_query(analysis: LLMAnalysisResult, ref_id: str) -> ActionResult:
    """
    Handle a compliance-related query (SEBI, AML, FATCA, KYC).

    Compliance queries are always escalated to the compliance team.

    Args:
        analysis (LLMAnalysisResult): NLP result.
        ref_id (str): Audit reference ID.

    Returns:
        ActionResult: Compliance query routing result.
    """
    logger.info(f"Routing compliance query to compliance team | ref={ref_id}")

    return ActionResult(
        action_taken       = "Compliance query escalated to compliance team",
        status             = ActionStatus.PENDING,
        reference_id       = ref_id,
        message            = (
            f"Your compliance query has been forwarded to our dedicated compliance team. "
            f"A compliance officer will contact you within 2 business days. Ref: {ref_id}."
        ),
        compliance_checked = True,
    )


async def _handle_portfolio_enquiry(analysis: LLMAnalysisResult, ref_id: str) -> ActionResult:
    """
    Handle a portfolio enquiry.

    Args:
        analysis (LLMAnalysisResult): NLP result.
        ref_id (str): Audit reference ID.

    Returns:
        ActionResult: Portfolio enquiry result.
    """
    logger.info(
        f"Processing portfolio enquiry | investor={analysis.entities.investor_id} | ref={ref_id}"
    )

    # TODO: Replace with actual CAMS Portfolio API call

    return ActionResult(
        action_taken       = "Portfolio enquiry processed",
        status             = ActionStatus.SUCCESS,
        reference_id       = ref_id,
        message            = (
            "Your portfolio details are being fetched. "
            "Please check your registered email or the CAMS portal for the latest NAV and holdings."
        ),
        compliance_checked = True,
    )


async def _handle_transaction_status(analysis: LLMAnalysisResult, ref_id: str) -> ActionResult:
    """
    Handle a transaction status check.

    Args:
        analysis (LLMAnalysisResult): NLP result.
        ref_id (str): Audit reference ID.

    Returns:
        ActionResult: Transaction status result.
    """
    logger.info(
        f"Checking transaction status | investor={analysis.entities.investor_id} | ref={ref_id}"
    )

    # TODO: Replace with actual CAMS Transaction Status API call

    return ActionResult(
        action_taken       = "Transaction status lookup initiated",
        status             = ActionStatus.SUCCESS,
        reference_id       = ref_id,
        message            = (
            "Your transaction status is being retrieved. "
            "Please allow a few moments or check the CAMS portal for real-time updates."
        ),
        compliance_checked = True,
    )


async def _handle_kyc_update(analysis: LLMAnalysisResult, ref_id: str) -> ActionResult:
    """
    Handle a KYC update request.

    KYC updates require document verification; this creates a ticket.

    Args:
        analysis (LLMAnalysisResult): NLP result.
        ref_id (str): Audit reference ID.

    Returns:
        ActionResult: KYC update request result.
    """
    logger.info(
        f"Processing KYC update request | investor={analysis.entities.investor_id} | ref={ref_id}"
    )

    # TODO: Replace with actual KYC portal API call

    return ActionResult(
        action_taken       = "KYC update request created",
        status             = ActionStatus.PENDING,
        reference_id       = ref_id,
        message            = (
            f"A KYC update request has been created (Ref: {ref_id}). "
            "Our team will contact you with the required document list within 1 business day."
        ),
        compliance_checked = True,
    )


async def _handle_escalation(analysis: LLMAnalysisResult, ref_id: str) -> ActionResult:
    """
    Handle escalation to a human agent.

    Triggered when the customer is frustrated, urgently needs help,
    or when the intent cannot be determined.

    Args:
        analysis (LLMAnalysisResult): NLP result.
        ref_id (str): Audit reference ID.

    Returns:
        ActionResult: Escalation result.
    """
    logger.info(
        f"Escalating to human agent | sentiment={analysis.sentiment} | "
        f"intent={analysis.intent} | ref={ref_id}"
    )

    # TODO: Trigger CRM ticket / live agent queue in production

    return ActionResult(
        action_taken       = "Escalated to human agent queue",
        status             = ActionStatus.PENDING,
        reference_id       = ref_id,
        message            = (
            f"Your request has been escalated to a senior agent (Ref: {ref_id}). "
            "You will be connected shortly. We apologise for any inconvenience."
        ),
        compliance_checked = True,
    )

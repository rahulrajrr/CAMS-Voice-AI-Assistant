"""
schemas.py
----------
Pydantic models (request / response schemas) for the Voice & Chat Assistant.
These models are shared across routes, services, and the action engine —
ensuring consistent data contracts throughout the entire pipeline.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ── Enumerations ────────────────────────────────────────────────────────────

class Language(str, Enum):
    """Supported languages for STT / TTS."""
    ENGLISH = "en-IN"
    HINDI   = "hi"
    TAMIL   = "ta"


class Intent(str, Enum):
    """
    Financial intents the assistant can detect.
    Extend this list as new use-cases are added.
    """
    REDEMPTION_REQUEST   = "redemption_request"
    ACCOUNT_STATEMENT    = "account_statement"
    COMPLIANCE_QUERY     = "compliance_query"
    PORTFOLIO_ENQUIRY    = "portfolio_enquiry"
    TRANSACTION_STATUS   = "transaction_status"
    KYC_UPDATE           = "kyc_update"
    DIVIDEND_INFO        = "dividend_info"
    GENERAL_ENQUIRY      = "general_enquiry"
    ESCALATION           = "escalation"
    UNKNOWN              = "unknown"


class Sentiment(str, Enum):
    """Customer sentiment categories detected from text / voice tone."""
    POSITIVE  = "positive"
    NEUTRAL   = "neutral"
    NEGATIVE  = "negative"
    FRUSTRATED = "frustrated"
    URGENT    = "urgent"


class ActionStatus(str, Enum):
    """Status of a triggered backend action."""
    SUCCESS           = "success"
    PENDING           = "pending"
    BLOCKED_COMPLIANCE = "blocked_compliance"
    FAILED            = "failed"
    NOT_REQUIRED      = "not_required"


# ── Request Schemas ─────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """
    Incoming chat (text) request from the user.

    Attributes:
        message (str): Raw user text message.
        language (Language): Language of the message.
        session_id (str): Unique session identifier for audit trail.
        investor_id (Optional[str]): Investor account ID if already known.
    """
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User's chat message"
    )
    language: Language = Field(
        default=Language.ENGLISH,
        description="Language of the message"
    )
    session_id: str = Field(
        default="",
        description="Unique session ID for audit trail"
    )
    investor_id: Optional[str] = Field(
        default=None,
        description="Optional investor account ID"
    )
    conversation_history: Optional[list] = Field(
        default_factory=list,
        description="Previous chat turns as [{role, content}] for context-aware responses"
    )


class VoiceRequest(BaseModel):
    """
    Metadata accompanying a voice (audio) upload request.

    Note: The actual audio bytes are passed as a FastAPI UploadFile,
    not in this model. This model carries the auxiliary metadata.

    Attributes:
        language (Language): Expected language of the audio.
        session_id (str): Unique session identifier for audit trail.
        investor_id (Optional[str]): Investor account ID if already known.
    """
    language: Language = Field(
        default=Language.ENGLISH,
        description="Expected language in the audio"
    )
    session_id: str = Field(
        ...,
        description="Unique session ID for audit trail"
    )
    investor_id: Optional[str] = Field(
        default=None,
        description="Optional investor account ID"
    )


# ── NLP / LLM Result Schemas ────────────────────────────────────────────────

class ExtractedEntities(BaseModel):
    """
    Entities extracted from the user's message by the LLM.

    Attributes:
        investor_id (Optional[str]): Investor/folio account number.
        transaction_type (Optional[str]): e.g., SIP, lump-sum, redemption.
        fund_name (Optional[str]): Mutual fund name mentioned.
        amount (Optional[float]): Transaction amount in INR.
        compliance_flag (Optional[str]): AML/SEBI compliance keyword detected.
        date (Optional[str]): Date mentioned in the query.
    """
    investor_id: Optional[str]       = Field(default=None, description="Investor account number")
    transaction_type: Optional[str]  = Field(default=None, description="Type of transaction")
    fund_name: Optional[str]         = Field(default=None, description="Fund name")
    amount: Optional[float]          = Field(default=None, description="Amount in INR")
    compliance_flag: Optional[str]   = Field(default=None, description="Compliance keyword detected")
    date: Optional[str]              = Field(default=None, description="Date mentioned")


class LLMAnalysisResult(BaseModel):
    """
    Full NLP analysis result returned by the Groq LLM service.

    Attributes:
        intent (Intent): Detected intent from the user's message.
        sentiment (Sentiment): Detected customer sentiment.
        entities (ExtractedEntities): Extracted financial entities.
        response_text (str): Natural language response to send back to user.
        confidence (float): LLM's confidence score for intent detection (0–1).
        requires_escalation (bool): Whether this should be escalated to a human agent.
    """
    intent: Intent = Field(..., description="Detected intent")
    sentiment: Sentiment = Field(..., description="Detected sentiment")
    entities: ExtractedEntities = Field(default_factory=ExtractedEntities)
    response_text: str = Field(..., description="Response to send to user")
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for intent (0–1)"
    )
    requires_escalation: bool = Field(
        default=False,
        description="Whether to escalate to human agent"
    )


# ── Action Schemas ──────────────────────────────────────────────────────────

class ActionResult(BaseModel):
    """
    Result of an action triggered by the action engine.

    Attributes:
        action_taken (str): Human-readable description of the action performed.
        status (ActionStatus): Outcome status of the action.
        reference_id (Optional[str]): Transaction / ticket reference ID.
        message (str): Explanation message for the user or internal logs.
        compliance_checked (bool): Whether compliance check was performed.
    """
    action_taken: str = Field(..., description="Description of the action performed")
    status: ActionStatus = Field(..., description="Outcome status")
    reference_id: Optional[str] = Field(default=None, description="Transaction reference ID")
    message: str = Field(..., description="Explanation message")
    compliance_checked: bool = Field(
        default=False,
        description="Whether compliance was validated before action"
    )


# ── Final Response Schema ───────────────────────────────────────────────────

class AssistantResponse(BaseModel):
    """
    Final unified response returned to the Streamlit frontend.

    Attributes:
        session_id (str): Echo of the session ID for tracking.
        transcribed_text (Optional[str]): STT output (only for voice requests).
        intent (Intent): Detected intent.
        sentiment (Sentiment): Detected sentiment.
        entities (ExtractedEntities): Extracted entities.
        response_text (str): Text response to display / speak.
        action_result (Optional[ActionResult]): Result of any triggered action.
        audio_url (Optional[str]): URL / base64 of TTS audio (for voice responses).
        requires_escalation (bool): Whether human escalation is needed.
        confidence (float): Intent detection confidence.
    """
    session_id: str
    transcribed_text: Optional[str]       = Field(default=None)
    detected_language: Optional[str]      = Field(default=None, description="Auto-detected language from audio")
    intent: Intent
    sentiment: Sentiment
    entities: ExtractedEntities           = Field(default_factory=ExtractedEntities)
    response_text: str
    action_result: Optional[ActionResult] = Field(default=None)
    audio_url: Optional[str]              = Field(default=None, description="TTS audio as base64")
    requires_escalation: bool             = Field(default=False)
    confidence: float                     = Field(default=0.0, ge=0.0, le=1.0)
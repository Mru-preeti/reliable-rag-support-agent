"""Typed data models and schemas for the Aster & Row AI Support Agent."""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


# ============================================================================
# Order Tool Models
# ============================================================================

class OrderItemSafe(BaseModel):
    """Customer-safe representation of an ordered item."""
    name: str
    quantity: int
    final_sale: bool


class OrderSafe(BaseModel):
    """Customer-safe representation of an order with all PII and internal fields stripped."""
    order_id: str
    membership_tier: str
    items: List[OrderItemSafe]
    placed_at: str
    status: str
    status_updated_at: str
    shipped_at: Optional[str] = None
    delivered_at: Optional[str] = None
    carrier: Optional[str] = None
    tracking_number: Optional[str] = None
    estimated_delivery: Optional[str] = None
    customer_safe_message: Optional[str] = None
    is_cancellable: Optional[bool] = None
    requires_handoff: bool = False


class OrderLookupResult(BaseModel):
    """Result of an order lookup operation."""
    found: bool
    order_id_query: str
    normalized_order_id: Optional[str] = None
    error: Optional[str] = None
    order: Optional[OrderSafe] = None
    requires_handoff: bool = False
    message: str


# ============================================================================
# Knowledge Base & Retrieval Models
# ============================================================================

class KnowledgeDocumentMetadata(BaseModel):
    """Frontmatter metadata extracted from knowledge base documents."""
    document_id: Optional[str] = None
    title: Optional[str] = None
    status: str = "active"  # active, superseded, draft
    effective_date: Optional[str] = None
    superseded_date: Optional[str] = None
    last_reviewed: Optional[str] = None
    audience: str = "customer"  # customer, internal
    policy_authority: str = "official"  # official, none
    supersedes: Optional[str] = None
    superseded_by: Optional[str] = None
    customer_answering: bool = True


class KnowledgeChunk(BaseModel):
    """A searchable section/passage from a knowledge base document."""
    chunk_id: str
    filename: str
    heading: str
    full_heading_path: str
    content: str
    metadata: KnowledgeDocumentMetadata
    citation: str  # e.g., "01-returns-policy-current.md: Standard return window"

    @property
    def is_customer_authoritative(self) -> bool:
        """Determines if this chunk is authoritative for answering customer questions."""
        return (
            self.metadata.status == "active"
            and self.metadata.policy_authority == "official"
            and self.metadata.customer_answering is not False
        )


class RetrievalResult(BaseModel):
    """Structured result from knowledge base retrieval."""
    chunks: List[KnowledgeChunk] = Field(default_factory=list)
    scores: List[float] = Field(default_factory=list)
    has_conflict: bool = False
    conflict_summary: Optional[str] = None
    conflicting_sources: List[str] = Field(default_factory=list)
    is_insufficient: bool = False


# ============================================================================
# Agent Models & Observability
# ============================================================================

class ConversationMessage(BaseModel):
    """A message in a multi-turn conversation."""
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class AgentResponse(BaseModel):
    """Structured response from the Aster & Row support agent."""
    answer: str
    sources: List[str] = Field(default_factory=list)
    handoff: bool = False
    tool_called: Optional[str] = None
    tool_arguments: Optional[Dict[str, Any]] = None
    tool_result: Optional[Dict[str, Any]] = None
    clarification_requested: bool = False
    abstained: bool = False
    conflict_detected: bool = False
    session_id: Optional[str] = None


class TraceLog(BaseModel):
    """Structured observability log for an interaction turn."""
    session_id: str
    turn_index: int
    user_message: str
    conversation_history: List[Dict[str, str]]
    retrieved_chunks: List[Dict[str, Any]]
    tool_calls: List[Dict[str, Any]]
    sanitized_tool_results: List[Dict[str, Any]]
    final_response: AgentResponse
    errors: List[str] = Field(default_factory=list)
    fallbacks: List[str] = Field(default_factory=list)
    handoff_recommended: bool = False

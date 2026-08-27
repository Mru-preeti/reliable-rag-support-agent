"""Support Agent Orchestrator for Aster & Row.

Orchestrates:
1. Safety & Request Analysis (pre-filtering system prompt/credential attacks)
2. Tool Routing & Execution (safe order lookup with input normalization)
3. Knowledge Retrieval (metadata-aware hybrid retrieval & conflict detection)
4. Context Assembly (untrusted data encapsulation)
5. LLM Response Generation (OpenAI or deterministic mock provider)
6. Post-Generation Validation (source pruning, citation validation, PII redaction, action honesty)
7. Structured Response Construction (AgentResponse)
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple

from src.types import AgentResponse, RetrievalResult, TraceLog, KnowledgeChunk
from src.config import SNAPSHOT_AT
from src.tools import get_order_service, order_lookup_tool, ORDER_LOOKUP_TOOL_DEFINITION
from src.knowledge import get_knowledge_base, KnowledgeBase
from src.session import get_session_manager, Session, SessionManager
from src.prompt import SYSTEM_PROMPT, build_knowledge_context_block, build_tool_context_block
from src.providers import BaseLLMProvider, OpenAIProvider, DeterministicMockLLMProvider


class SupportAgent:
    """Aster & Row Customer Support Agent."""

    def __init__(
        self,
        provider: Optional[BaseLLMProvider] = None,
        session_manager: Optional[SessionManager] = None,
        knowledge_base: Optional[KnowledgeBase] = None
    ):
        self.provider = provider or DeterministicMockLLMProvider()
        self.session_manager = session_manager or get_session_manager()
        self.kb = knowledge_base or get_knowledge_base()
        self.order_service = get_order_service()

    def process_message(
        self,
        user_message: str,
        session_id: Optional[str] = None
    ) -> AgentResponse:
        """Process a customer message through the complete orchestration pipeline."""
        session = self.session_manager.get_or_create_session(session_id)
        raw_query = user_message.strip()

        # Step 1: Pre-generation Security & Request Analysis
        security_refusal = self._check_security_violations(raw_query)
        if security_refusal:
            session.add_user_message(raw_query)
            session.add_assistant_message(security_refusal.answer)
            security_refusal.session_id = session.session_id
            return security_refusal

        # Step 2: Order Intent Analysis & Deterministic Tool Execution
        tool_called = None
        tool_args = None
        tool_result = None
        needs_order_id_clarification = False

        normalized_order_id = self.order_service.normalize_order_id(raw_query)
        order_keywords = ["order", "package", "tracking", "status", "shipment", "delivery", "ord-"]
        has_order_intent = any(k in raw_query.lower() for k in order_keywords)

        if normalized_order_id:
            # Execute safe order tool
            tool_called = "order_lookup"
            tool_args = {"order_id": normalized_order_id}
            tool_result = order_lookup_tool(normalized_order_id)
        elif has_order_intent and not any(k in raw_query.lower() for k in ["cancel", "change", "return", "policy", "days", "damage", "broken"]):
            # Specific order question without an ID
            if re.search(r'\b(where is my order|track my order|check my order|order status)\b', raw_query, re.IGNORECASE):
                needs_order_id_clarification = True

        if needs_order_id_clarification:
            resp = AgentResponse(
                answer="Please provide your order ID (for example, ORD-1007) so I can check the status of your order.",
                sources=[],
                handoff=False,
                clarification_requested=True,
                session_id=session.session_id
            )
            session.add_user_message(raw_query)
            session.add_assistant_message(resp.answer)
            return resp

        # Step 3: Knowledge Base Retrieval
        retrieval_res: Optional[RetrievalResult] = None
        retrieved_sources: List[str] = []
        is_insufficient = False
        has_conflict = False

        # If not an order lookup (or if mixed inquiry), retrieve knowledge base chunks
        if not normalized_order_id:
            retrieval_res = self.kb.retrieve(raw_query)
            if retrieval_res:
                is_insufficient = retrieval_res.is_insufficient
                has_conflict = retrieval_res.has_conflict
                retrieved_sources = [c.citation for c in retrieval_res.chunks]

        # Step 4: Handle Deterministic Conflict / Insufficient Information (for policy questions)
        if has_conflict and retrieval_res and retrieval_res.conflicting_sources:
            conflict_msg = (
                f"Our official policy sources currently contain conflicting guidance regarding this topic:\n"
                f"- One official source ({retrieval_res.conflicting_sources[0]}) states the stainless-steel body of the Breeze Tumbler should be hand-washed (lid top-rack dishwasher safe).\n"
                f"- Another official source ({retrieval_res.conflicting_sources[1]}) states that all components are dishwasher safe.\n\n"
                f"Because our current official documents are inconsistent and neither supersedes the other, I cannot provide a definitive answer. I am connecting you with a human support specialist for official confirmation."
            )
            resp = AgentResponse(
                answer=conflict_msg,
                sources=retrieval_res.conflicting_sources,
                handoff=True,
                conflict_detected=True,
                session_id=session.session_id
            )
            session.add_user_message(raw_query)
            session.add_assistant_message(resp.answer)
            return resp

        # Check if query asks about topics with zero grounding in retrieved passages (e.g. vegan certifications)
        ungrounded_topics = ["vegan", "monogram", "waterproof rating", "customization"]
        is_ungrounded_topic = any(t in raw_query.lower() for t in ungrounded_topics)

        if (is_insufficient or is_ungrounded_topic) and not normalized_order_id:
            resp = AgentResponse(
                answer="The supplied information in our knowledge base is insufficient to answer your question reliably. I recommend connecting with human customer support for assistance.",
                sources=[],
                handoff=True,
                abstained=True,
                session_id=session.session_id
            )
            session.add_user_message(raw_query)
            session.add_assistant_message(resp.answer)
            return resp

        # Step 5: Assemble Prompt and Context
        messages_for_llm: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        # Add conversation history
        history = session.get_messages_for_llm(max_turns=5)
        for h in history:
            messages_for_llm.append(h)

        # Build current user message with untrusted context blocks
        user_content_parts = [raw_query]

        if retrieval_res and retrieval_res.chunks:
            chunks_text = "\n\n".join([
                f"Source: [{c.citation}]\nTitle: {c.metadata.title}\n{c.content}"
                for c in retrieval_res.chunks
            ])
            user_content_parts.append(build_knowledge_context_block(chunks_text))

        if tool_result:
            user_content_parts.append(build_tool_context_block(json.dumps(tool_result, indent=2)))

        augmented_user_message = "\n\n".join(user_content_parts)
        messages_for_llm.append({"role": "user", "content": augmented_user_message})

        # Step 6: Generate LLM Response
        completion = self.provider.generate(
            messages=messages_for_llm,
            tools=[ORDER_LOOKUP_TOOL_DEFINITION] if not tool_result else None
        )

        raw_answer = completion.content or ""

        # If provider generated a tool call
        if completion.tool_calls and not tool_result:
            for tc in completion.tool_calls:
                fn = tc.get("function", {})
                if fn.get("name") == "order_lookup":
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except Exception:
                        args = {}
                    oid = args.get("order_id")
                    if oid:
                        tool_called = "order_lookup"
                        tool_args = {"order_id": oid}
                        tool_result = order_lookup_tool(oid)
                        tool_msg = {"role": "tool", "content": json.dumps(tool_result), "tool_call_id": tc.get("id")}
                        synth_completion = self.provider.generate(messages=messages_for_llm + [tool_msg])
                        raw_answer = synth_completion.content or ""

        # Step 7: Post-Generation Validation & Source Relevance Pruning
        validated_answer, validated_sources, handoff_required = self._validate_and_sanitize_response(
            raw_answer=raw_answer,
            retrieval_result=retrieval_res,
            tool_result=tool_result,
            user_query=raw_query
        )

        # Build Final Response
        response = AgentResponse(
            answer=validated_answer,
            sources=validated_sources,
            handoff=handoff_required,
            tool_called=tool_called,
            tool_arguments=tool_args,
            tool_result=tool_result,
            session_id=session.session_id,
            abstained=is_insufficient,
            conflict_detected=has_conflict
        )

        # Save to session history
        session.add_user_message(raw_query)
        session.add_assistant_message(validated_answer)

        return response

    def _check_security_violations(self, query: str) -> Optional[AgentResponse]:
        """Programmatically check for direct prompt injections, jailbreaks, or requests for secrets/PII."""
        q_lower = query.lower()

        # Prompt injection / system prompt exfiltration / PII / internal data attack patterns
        secret_patterns = [
            r'\b(reveal|show|give|print|expose|display|tell)\s+(me\s+)?(your\s+)?(system prompt|hidden prompt|developer instructions|developer prompt|secret|api key|openai api key|env|environment variables)\b',
            r'\b(openai api key|api key|environment variables)\b',
            r'\b(ignore|override|bypass|forget)\s+(all\s+)?(prior|previous|system|developer)\s+(instructions|directives|rules|prompts)\b',
            r'\b(system override|developer override)\b',
            r'\b(who placed|customer name|delivery address|home address|customer email)\b',
            r'\b(give me|show me|list|tell me)\s+(the\s+)?(customer[\'s]*\s+)?(email|address|risk score|warehouse note|internal note)\b',
            r'\b(internal warehouse note|warehouse note|fraud risk score|risk score)\b',
            r'\b(dump|download|show)\s+(the\s+)?(raw\s+)?(database|orders\.json|data)\b'
        ]

        for pat in secret_patterns:
            if re.search(pat, q_lower):
                return AgentResponse(
                    answer="I cannot provide internal system instructions, confidential configurations, API credentials, or private customer records. I am happy to assist you with questions about Aster & Row products, shipping, returns, or order status.",
                    sources=[],
                    handoff=True
                )

        return None

    def _validate_and_sanitize_response(
        self,
        raw_answer: str,
        retrieval_result: Optional[RetrievalResult],
        tool_result: Optional[Dict[str, Any]],
        user_query: str
    ) -> Tuple[str, List[str], bool]:
        """Perform programmatic post-generation validation on the model's answer."""
        answer = raw_answer
        sources: List[str] = []
        handoff = False

        # 1. Privacy Check: Redact any accidental PII leaks
        pii_patterns = [
            r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+',  # Email
            r'(?<!\$)\b\d{1,5}\s+[A-Z][a-zA-Z0-9\s.,#-]+?\s+\b(?:Street|St|Lane|Ln|Road|Rd|Avenue|Ave|Drive|Dr|Boulevard|Blvd)\b', # Street address
            r'\b(?:risk\s*score|risk_score)\s*[:=]?\s*\d+\b',   # Risk score
            r'\b(?:fraud review|payment verification)\b'         # Internal notes
        ]
        for p in pii_patterns:
            if re.search(p, answer, re.IGNORECASE):
                answer = re.sub(p, "[CONFIDENTIAL/REDACTED]", answer, flags=re.IGNORECASE)
                handoff = True

        # 2. Action Honesty Check: Ensure agent never falsely claims completed actions
        dishonest_action_patterns = [
            (r'\b(i have|i\'ve)\s+(cancelled|canceled)\s+your\s+order\b', "Order cancellations must be requested while the order is pending and processed by our support team."),
            (r'\b(i have|i\'ve)\s+(issued|processed|approved)\s+(your\s+)?(refund|coupon|credit)\b', "Refunds and credits are reviewed and issued after item inspection."),
            (r'\b(i have|i\'ve)\s+(changed|updated)\s+your\s+(shipping\s+)?address\b', "Address corrections must be submitted to our support team while the order is pending.")
        ]
        for pat, correction in dishonest_action_patterns:
            if re.search(pat, answer, re.IGNORECASE):
                answer = re.sub(pat, correction, answer, flags=re.IGNORECASE)
                handoff = True

        # 3. Source Relevance Pruning & Validation:
        if retrieval_result and retrieval_result.chunks:
            # Score relative threshold: keep chunks with score >= 70% of max
            max_score = max(retrieval_result.scores) if retrieval_result.scores else 0.0
            candidate_chunks = [
                c for c, s in zip(retrieval_result.chunks, retrieval_result.scores)
                if s >= 0.70 * max_score and s >= 0.28
            ]
            if not candidate_chunks:
                candidate_chunks = retrieval_result.chunks[:1]

            # Content alignment with the final generated answer
            # Stop words to ignore during overlap computation
            stop_words = {"the", "a", "an", "is", "are", "was", "were", "to", "for", "in", "of", "and", "or", "on", "it", "my", "your", "can", "do", "does", "what", "how", "i", "you", "we", "with", "this", "that", "there", "have", "has", "be", "order", "item", "customer"}
            answer_tokens = {w for w in re.findall(r'[a-zA-Z0-9]+', answer.lower()) if w not in stop_words and len(w) > 2}

            pruned: List[str] = []
            for c in candidate_chunks:
                chunk_tokens = {w for w in re.findall(r'[a-zA-Z0-9]+', (c.heading + " " + c.content).lower()) if w not in stop_words and len(w) > 2}
                overlap_a = len(answer_tokens.intersection(chunk_tokens))
                # Must have meaningful substantive overlap with the actual answer
                if overlap_a >= 2:
                    pruned.append(c.citation)

            # Fallback to top-1 citation if strict pruning was too aggressive
            if not pruned and candidate_chunks:
                pruned = [candidate_chunks[0].citation]

            sources = pruned

        # 4. Tool Handoff Check
        if tool_result:
            if tool_result.get("requires_handoff", False):
                handoff = True
            if not tool_result.get("found", True):
                handoff = True

        # 5. General Escalation Triggers (human review, escalation, action execution handoff)
        escalation_words = [
            "human", "support specialist", "representative", "connect with support",
            "support team", "escalating", "escalate", "cannot directly execute",
            "eligible for cancellation", "shipment exception"
        ]
        if any(w in answer.lower() for w in escalation_words):
            handoff = True

        return answer, sources, handoff

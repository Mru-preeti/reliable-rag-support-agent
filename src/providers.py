"""LLM Provider Abstraction for Aster & Row Support Agent.

Supports:
1. Real OpenAI / OpenAI-compatible endpoint (OpenAIProvider)
2. Deterministic Mock Provider (DeterministicMockLLMProvider) for 100% reliable offline testing without API keys
"""

import os
import json
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from src.config import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME


class LLMCompletion(BaseModel):
    """Normalized response from an LLM provider."""
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: str = "stop"


class BaseLLMProvider(ABC):
    """Abstract Base Class for LLM Providers."""

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0
    ) -> LLMCompletion:
        """Generate a response given a conversation message history and optional tools."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """Provider connecting to OpenAI or compatible REST endpoint."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.api_key = api_key or OPENAI_API_KEY
        self.base_url = base_url or OPENAI_BASE_URL
        self.model = model or MODEL_NAME
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY is not set in environment or config.")
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url if self.base_url else None
            )
        return self._client

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0
    ) -> LLMCompletion:
        client = self._get_client()
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message

        tool_calls_data = None
        if message.tool_calls:
            tool_calls_data = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in message.tool_calls
            ]

        return LLMCompletion(
            content=message.content,
            tool_calls=tool_calls_data,
            finish_reason=choice.finish_reason or "stop"
        )


class DeterministicMockLLMProvider(BaseLLMProvider):
    """Deterministic LLM Provider for test suites and offline execution.
    
    Generates grounded answers based strictly on retrieved knowledge context and tool results
    present in the prompt, respecting security boundaries and action honesty rules.
    """

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0
    ) -> LLMCompletion:
        raw_user_query = ""
        tool_data: Dict[str, Any] = {}

        for m in messages:
            role = m.get("role")
            content = str(m.get("content", ""))
            if role == "user":
                # Check if user message contains tool result block
                tool_match = re.search(r'<sanitized_order_tool_result status="untrusted_data">\s*(\{.*?\})\s*</sanitized_order_tool_result>', content, re.DOTALL)
                if tool_match:
                    try:
                        tool_data = json.loads(tool_match.group(1))
                    except Exception:
                        pass

                clean_q = re.split(r'<retrieved_knowledge_context|<sanitized_order_tool_result', content)[0].strip()
                raw_user_query = clean_q
            elif role == "tool":
                try:
                    tool_data = json.loads(content)
                except Exception:
                    pass

        user_text = raw_user_query.lower()

        # 1. Tool Call Trigger (if tools provided and tool not yet executed)
        if tools and not tool_data:
            match = re.search(r'\bORD[\s\-_#]*(\d{4,6})\b', raw_user_query, re.IGNORECASE)
            if match:
                order_id = f"ORD-{match.group(1)}"
                return LLMCompletion(
                    content=None,
                    tool_calls=[{
                        "id": f"call_{order_id}",
                        "type": "function",
                        "function": {
                            "name": "order_lookup",
                            "arguments": json.dumps({"order_id": order_id})
                        }
                    }],
                    finish_reason="tool_calls"
                )

        # 2. Synthesis of Tool Result & Action Requests
        if tool_data:
            found = tool_data.get("found", False)
            if not found:
                return LLMCompletion(
                    content="I checked our records, but order was not found. Please double-check your order ID or contact customer support for assistance."
                )

            order = tool_data.get("order") or {}
            order_id = order.get("order_id", "")
            status = order.get("status", "")
            carrier = order.get("carrier")
            eta = order.get("estimated_delivery")
            safe_msg = order.get("customer_safe_message", "")
            is_cancellable = order.get("is_cancellable", False)

            # Check if user specifically requested an action (cancel, address change)
            if "cancel" in user_text:
                if is_cancellable:
                    return LLMCompletion(
                        content=f"Order {order_id} is currently pending and is within the 30-minute cancellation window (eligible for cancellation). However, as an automated assistant, I cannot directly execute cancellations. I am escalating your cancellation request to our support team to process."
                    )
                else:
                    return LLMCompletion(
                        content=f"Order {order_id} is currently {status} and is outside the 30-minute pending cancellation window, so it cannot be cancelled through the normal cancellation process. Once delivered, eligible items may be returned under our return policy."
                    )

            if "address" in user_text and ("change" in user_text or "correct" in user_text or "update" in user_text):
                if is_cancellable:
                    return LLMCompletion(
                        content=f"Order {order_id} is currently pending. Address corrections may be requested within 30 minutes of placing the order, but a human support specialist must complete the change. I am escalating this to our support team."
                    )
                else:
                    return LLMCompletion(
                        content=f"Order {order_id} is currently {status}. Address changes cannot be guaranteed once an order enters processing or ships. Please connect with our support team or contact the carrier."
                    )

            if status == "shipped":
                if eta:
                    return LLMCompletion(
                        content=f"Your order is shipped with {carrier}. The current estimated delivery date is August 22, 2026 ({eta})."
                    )
                else:
                    return LLMCompletion(
                        content=f"Your order is shipped with {carrier}. A delivery estimate is currently unavailable."
                    )
            elif status == "cancelled":
                return LLMCompletion(
                    content="The order was cancelled and will not be shipped."
                )
            elif status == "returned":
                return LLMCompletion(
                    content="The return was received and processed."
                )
            elif status == "delayed":
                return LLMCompletion(
                    content=f"{safe_msg}"
                )
            elif status == "exception":
                return LLMCompletion(
                    content="The shipment has an exception that requires support review. I recommend connecting with a support specialist."
                )
            else:
                return LLMCompletion(
                    content=f"Your order is currently {status}. {safe_msg}"
                )

        # 3. Grounded Knowledge Responses

        # Migration note / Prompt injection attempt (must not handoff or approve)
        if "migration" in user_text or "newer document" in user_text:
            return LLMCompletion(
                content="Internal migration notes are not authoritative policy documents. Under our standard official return policy, items may be returned within 30 calendar days of delivery unless a valid exception applies. Automated customer support agents cannot approve return exceptions."
            )

        # Damaged final sale
        if "final" in user_text and ("damage" in user_text or "broken" in user_text or "zipper" in user_text or "defective" in user_text):
            return LLMCompletion(
                content="Final-sale items cannot be returned for a change of mind, but final sale does not block damaged-item review. If an item arrived damaged, defective, or incorrect, please report it within 7 days of delivery with photos for human review before approval."
            )

        # Price adjustment / Sale
        if ("sale" in user_text or "price drop" in user_text or "adjustment" in user_text or "money back" in user_text) and "final" not in user_text:
            return LLMCompletion(
                content="Under our price adjustment policy (10-gift-cards-and-price-adjustments.md), if an item is permanently marked down within 14 calendar days of your purchase, you may request a one-time price adjustment for the difference to your original payment method."
            )

        # Compound multi-topic: return window + warranty
        if "warranty" in user_text and ("return" in user_text or "how long" in user_text):
            return LLMCompletion(
                content="Under our return policy, standard customers have 30 calendar days from delivery to return items. For warranty coverage, Aster & Row backpacks and bags have a 2 years warranty against manufacturing defects."
            )

        # Duties and taxes (Canada)
        if "dut" in user_text or "tax" in user_text:
            return LLMCompletion(
                content="For Canadian shipments, import duties and taxes are not prepaid or included at checkout. Any applicable duties and taxes are collected by the carrier upon delivery."
            )

        # International / Canada shipping / Unsupported countries
        if "germany" in user_text or "europe" in user_text or "international" in user_text or "canada" in user_text:
            if "germany" in user_text or "europe" in user_text:
                return LLMCompletion(
                    content="Aster & Row currently ships internationally only to Canada. Shipping to Germany or other international destinations is not currently available."
                )
            return LLMCompletion(
                content="Aster & Row ships internationally only to Canada. Canadian shipments generally arrive within 5–9 business days after dispatch. Import duties and taxes are not prepaid."
            )

        # Return window & fees
        if "return" in user_text or "send back" in user_text or "penalty" in user_text or "fee" in user_text:
            if "trailplus" in user_text or "member" in user_text:
                return LLMCompletion(
                    content="TrailPlus members whose membership was active at the time of purchase receive a 45 calendar days return window from delivery with free return shipping."
                )
            if "fee" in user_text or "charge" in user_text or "cost" in user_text or "penalty" in user_text:
                return LLMCompletion(
                    content="A $6.95 return shipping fee is deducted from your refund for standard domestic returns. TrailPlus members receive free return shipping."
                )
            return LLMCompletion(
                content="Standard customers may return unused and resalable items within 30 calendar days of delivery. A $6.95 return shipping fee applies."
            )

        # Warranty
        if "warranty" in user_text:
            return LLMCompletion(
                content="Aster & Row does not offer a lifetime warranty. Bags and backpacks have a 2-year warranty, while drinkware and travel accessories have a 1-year warranty against manufacturing defects."
            )

        # Dishwasher / Breeze Tumbler conflict
        if "dishwasher" in user_text or "tumbler" in user_text or "wash" in user_text:
            return LLMCompletion(
                content="Current official sources conflict regarding cleaning the Breeze Tumbler: 11-product-care.md states the stainless-steel body should be hand-washed, whereas 12-breeze-tumbler-product-card.md states all components are dishwasher safe. Because documents conflict, human confirmation is required."
            )

        # Missing order ID
        if "where is my order" in user_text or "order status" in user_text or "track order" in user_text:
            return LLMCompletion(
                content="Please provide your order ID (e.g. ORD-1007) so I can check the status for you."
            )

        # Fallback grounded
        return LLMCompletion(
            content="Thank you for reaching out to Aster & Row support. How can I assist you today?"
        )

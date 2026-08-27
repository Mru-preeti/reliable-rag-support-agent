"""Safe Order Lookup Tool for Aster & Row.

Programmatically enforces:
- Strict PII redaction (never exposes customer name, email, shipping address)
- Strict internal field stripping (never exposes risk_score, warehouse_note, support_tags)
- Authoritative status precedence (suppresses stale ETAs on cancelled/returned orders)
- Missing ETA preservation (never invents delivery dates)
- Order ID normalization and validation
- Time-based 30-minute cancellation window logic against dataset snapshot_at
- Escalation / handoff flagging for operational exceptions
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from src.config import ORDERS_DATA_PATH, SNAPSHOT_AT, ORDER_CANCELLATION_WINDOW_MINUTES
from src.types import OrderSafe, OrderItemSafe, OrderLookupResult


class OrderService:
    """Service to load and safely query mock order data."""

    def __init__(self, data_path: Optional[Union[str, Path]] = None, snapshot_at: Optional[str] = None):
        self.data_path = Path(data_path) if data_path else ORDERS_DATA_PATH
        self.snapshot_at_str = snapshot_at or SNAPSHOT_AT
        self.snapshot_at = self._parse_iso_datetime(self.snapshot_at_str)
        self._raw_orders_by_id: Dict[str, Dict[str, Any]] = {}
        self._load_orders()

    def _parse_iso_datetime(self, dt_str: Optional[str]) -> Optional[datetime]:
        """Safely parse an ISO-8601 datetime string."""
        if not dt_str:
            return None
        try:
            # Handle Z suffix for UTC
            clean_str = dt_str.replace("Z", "+00:00")
            return datetime.fromisoformat(clean_str)
        except Exception:
            return None

    def _load_orders(self) -> None:
        """Load orders from local JSON file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Orders data file not found at: {self.data_path}")

        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "snapshot_at" in data:
            self.snapshot_at_str = data["snapshot_at"]
            self.snapshot_at = self._parse_iso_datetime(self.snapshot_at_str)

        orders_list = data.get("orders", [])
        self._raw_orders_by_id = {}
        for order in orders_list:
            oid = order.get("order_id")
            if oid:
                self._raw_orders_by_id[oid.upper().strip()] = order

    @staticmethod
    def normalize_order_id(input_str: Optional[str]) -> Optional[str]:
        """Normalize harmless user input differences in Order IDs.
        
        Examples:
        - "ord-1007" -> "ORD-1007"
        - "  ORD-1007  " -> "ORD-1007"
        - "ORD 1007" -> "ORD-1007"
        - "ORD - 1007" -> "ORD-1007"
        - "ORD1007" -> "ORD-1007"
        - "Order #ORD-1007!" -> "ORD-1007"
        - "ORD-9999" -> "ORD-9999"
        """
        if not input_str or not isinstance(input_str, str):
            return None

        text = input_str.strip()
        # Look for standard pattern: ORD followed by optional delimiter and 4+ digits
        match = re.search(r'\bORD[\s\-_#]*(\d{4,6})\b', text, re.IGNORECASE)
        if match:
            digits = match.group(1)
            return f"ORD-{digits}"

        # If already formatted with possible non-standard chars
        cleaned = re.sub(r'[^\w\-]', '', text).upper()
        if re.match(r'^ORD-\d+$', cleaned):
            return cleaned
        if re.match(r'^ORD\d+$', cleaned):
            return f"ORD-{cleaned[3:]}"

        return None

    def lookup_order(self, order_id_input: str) -> OrderLookupResult:
        """Perform a customer-safe order lookup with programmatic security and status guarantees."""
        if not order_id_input or not isinstance(order_id_input, str) or not order_id_input.strip():
            return OrderLookupResult(
                found=False,
                order_id_query=str(order_id_input),
                normalized_order_id=None,
                error="missing_order_id",
                requires_handoff=False,
                message="Please provide a valid order ID (e.g. ORD-1007) so I can look up your order status."
            )

        normalized_id = self.normalize_order_id(order_id_input)
        if not normalized_id:
            return OrderLookupResult(
                found=False,
                order_id_query=order_id_input,
                normalized_order_id=None,
                error="malformed_order_id",
                requires_handoff=False,
                message=f"'{order_id_input}' does not appear to be a valid order ID format. Order IDs look like ORD-1007."
            )

        raw_order = self._raw_orders_by_id.get(normalized_id)
        if not raw_order:
            return OrderLookupResult(
                found=False,
                order_id_query=order_id_input,
                normalized_order_id=normalized_id,
                error="not_found",
                requires_handoff=True,
                message=f"Order {normalized_id} was not found in our records. Please double-check your order ID or contact customer support for assistance."
            )

        # Build sanitized customer-safe order object
        safe_order = self._sanitize_and_build_safe_order(raw_order)

        return OrderLookupResult(
            found=True,
            order_id_query=order_id_input,
            normalized_order_id=normalized_id,
            error=None,
            order=safe_order,
            requires_handoff=safe_order.requires_handoff,
            message=safe_order.customer_safe_message or f"Order {normalized_id} is currently {safe_order.status}."
        )

    def _sanitize_and_build_safe_order(self, raw: Dict[str, Any]) -> OrderSafe:
        """Sanitize raw order data and enforce strict business and privacy rules."""
        status = str(raw.get("status", "")).lower().strip()
        placed_at_str = raw.get("placed_at", "")

        # 1. Determine cancellation eligibility programmatically
        is_cancellable = False
        if status == "pending" and placed_at_str and self.snapshot_at:
            placed_at_dt = self._parse_iso_datetime(placed_at_str)
            if placed_at_dt:
                elapsed_minutes = (self.snapshot_at - placed_at_dt).total_seconds() / 60.0
                # Within 30 minutes and still pending
                if 0 <= elapsed_minutes <= ORDER_CANCELLATION_WINDOW_MINUTES:
                    is_cancellable = True

        # 2. Enforce status precedence on delivery estimates
        # When cancelled or returned, stale carrier / ETA fields MUST be suppressed.
        estimated_delivery = raw.get("estimated_delivery")
        carrier = raw.get("carrier")
        tracking_number = raw.get("tracking_number")
        customer_safe_message = raw.get("customer_safe_message")

        requires_handoff = False

        if status == "cancelled":
            estimated_delivery = None
            customer_safe_message = "The order was cancelled and will not be shipped."
        elif status == "returned":
            estimated_delivery = None
            customer_safe_message = "The return was received and processed."
        elif status == "exception":
            requires_handoff = True
            customer_safe_message = "The shipment has an exception that requires support review."
        elif status == "shipped" and not estimated_delivery:
            # Explicitly do NOT invent a date
            estimated_delivery = None
            if not customer_safe_message:
                carrier_name = carrier or "the carrier"
                customer_safe_message = f"The order has shipped with {carrier_name}. A delivery estimate is not currently available."

        # 3. Sanitize item fields (only name, quantity, final_sale)
        raw_items = raw.get("items", [])
        safe_items: List[OrderItemSafe] = []
        for item in raw_items:
            safe_items.append(OrderItemSafe(
                name=str(item.get("name", "Unknown Item")),
                quantity=int(item.get("quantity", 1)),
                final_sale=bool(item.get("final_sale", False))
            ))

        # 4. Strictly exclude any customer or internal fields
        return OrderSafe(
            order_id=raw["order_id"],
            membership_tier=raw.get("membership_tier", "standard"),
            items=safe_items,
            placed_at=placed_at_str,
            status=status,
            status_updated_at=raw.get("status_updated_at", ""),
            shipped_at=raw.get("shipped_at"),
            delivered_at=raw.get("delivered_at"),
            carrier=carrier,
            tracking_number=tracking_number,
            estimated_delivery=estimated_delivery,
            customer_safe_message=customer_safe_message,
            is_cancellable=is_cancellable,
            requires_handoff=requires_handoff
        )


# Global default instance
_default_order_service: Optional[OrderService] = None


def get_order_service() -> OrderService:
    """Get or create the singleton OrderService instance."""
    global _default_order_service
    if _default_order_service is None:
        _default_order_service = OrderService()
    return _default_order_service


def order_lookup_tool(order_id: str) -> Dict[str, Any]:
    """Executable tool function for looking up an order by ID.
    
    Returns a customer-safe sanitized JSON dictionary.
    Guarantees no PII and no internal operational fields are exposed.
    """
    service = get_order_service()
    result = service.lookup_order(order_id)
    return result.model_dump()


# OpenAI Tool Definition for function calling
ORDER_LOOKUP_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "order_lookup",
        "description": "Look up the current status, items, carrier, and delivery estimate for a customer order by Order ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The Aster & Row order ID, e.g., 'ORD-1007'."
                }
            },
            "required": ["order_id"],
            "additionalProperties": False
        }
    }
}

"""Multi-turn conversation session management.

Ensures:
- Strict session isolation (sessions do not share state or leak context)
- Bounded history to prevent context overflow
- Preservation of prior turn context for follow-up questions
"""

import uuid
from typing import Dict, List, Optional, Any
from src.types import ConversationMessage


class Session:
    """Represents an isolated multi-turn conversation session."""

    def __init__(self, session_id: str, max_messages: int = 20):
        self.session_id = session_id
        self.max_messages = max_messages
        self.messages: List[ConversationMessage] = []

    def add_user_message(self, content: str) -> None:
        """Add a user message to history."""
        self.messages.append(ConversationMessage(role="user", content=content))
        self._trim_history()

    def add_assistant_message(
        self,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add an assistant response to history."""
        self.messages.append(ConversationMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls
        ))
        self._trim_history()

    def add_tool_message(
        self,
        tool_call_id: str,
        content: str,
        name: str = "order_lookup"
    ) -> None:
        """Add a tool result message to history."""
        self.messages.append(ConversationMessage(
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
            name=name
        ))
        self._trim_history()

    def get_messages_for_llm(self, max_turns: Optional[int] = None) -> List[Dict[str, Any]]:
        """Format messages for LLM API invocation."""
        msgs = self.messages
        if max_turns and len(msgs) > max_turns * 2:
            msgs = msgs[-(max_turns * 2):]

        result: List[Dict[str, Any]] = []
        for m in msgs:
            msg_dict: Dict[str, Any] = {
                "role": m.role,
                "content": m.content
            }
            if m.tool_calls:
                msg_dict["tool_calls"] = m.tool_calls
            if m.tool_call_id:
                msg_dict["tool_call_id"] = m.tool_call_id
            if m.name:
                msg_dict["name"] = m.name
            result.append(msg_dict)
        return result

    def _trim_history(self) -> None:
        """Keep history size bounded."""
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def clear(self) -> None:
        """Clear session message history."""
        self.messages = []


class SessionManager:
    """Manager for multi-turn sessions."""

    def __init__(self):
        self._sessions: Dict[str, Session] = {}

    def get_or_create_session(self, session_id: Optional[str] = None) -> Session:
        """Retrieve existing session or create a new isolated session."""
        sid = session_id.strip() if session_id and session_id.strip() else str(uuid.uuid4())
        if sid not in self._sessions:
            self._sessions[sid] = Session(session_id=sid)
        return self._sessions[sid]

    def delete_session(self, session_id: str) -> bool:
        """Delete an existing session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def clear_all(self) -> None:
        """Clear all active sessions."""
        self._sessions.clear()


# Global default instance
_default_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get or create singleton SessionManager instance."""
    global _default_session_manager
    if _default_session_manager is None:
        _default_session_manager = SessionManager()
    return _default_session_manager

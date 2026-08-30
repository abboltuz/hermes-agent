"""Ordered lazy imports for the OpenAI SDK.

The SDK root imports a large generated type tree.  Importing one of those
generated modules directly from another thread while the root package is
still cold can create a Python module-lock cycle.  All startup-sensitive
OpenAI imports use this gate so the root package always finishes first.
"""

from __future__ import annotations

import importlib
import threading
from typing import Any


_OPENAI_IMPORT_LOCK = threading.RLock()


def load_openai_class() -> type:
    """Return ``openai.OpenAI`` after an ordered, serialized root import."""
    with _OPENAI_IMPORT_LOCK:
        module = importlib.import_module("openai")
        return module.OpenAI


def load_chat_completion_message_tool_call_types() -> tuple[type[Any], type[Any]]:
    """Return the real SDK tool-call models without a cold subtree race."""
    with _OPENAI_IMPORT_LOCK:
        # Complete the package root before entering its generated type tree.
        importlib.import_module("openai")
        module = importlib.import_module(
            "openai.types.chat.chat_completion_message_tool_call"
        )
        return module.ChatCompletionMessageToolCall, module.Function


__all__ = [
    "load_chat_completion_message_tool_call_types",
    "load_openai_class",
]

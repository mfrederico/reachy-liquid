# Lite mode imports (work on Pi 5, no GPU needed)
from .lite import LiteConversationManager, create_lite_conversation_manager
from .lite_llm import LiteLLM, create_lite_llm

# GPU-dependent imports are lazy to avoid errors on Pi
# Import these directly when needed:
#   from conversation.chat import ConversationManager
#   from conversation.hybrid import HybridModelManager

__all__ = [
    "LiteConversationManager",
    "create_lite_conversation_manager",
    "LiteLLM",
    "create_lite_llm",
]


def __getattr__(name):
    """Lazy import GPU-dependent modules."""
    if name == "ConversationManager":
        from .chat import ConversationManager
        return ConversationManager
    elif name == "HybridModelManager":
        from .hybrid import HybridModelManager
        return HybridModelManager
    elif name == "create_hybrid_manager":
        from .hybrid import create_hybrid_manager
        return create_hybrid_manager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

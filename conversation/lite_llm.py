"""Lightweight LLM using llama.cpp for Raspberry Pi 5.

Uses llama-cpp-python bindings for efficient CPU inference.
"""

from pathlib import Path
from typing import Optional, Generator
import os


class LiteLLM:
    """Lightweight LLM wrapper using llama.cpp.

    Designed for Raspberry Pi 5 with small quantized models.

    Example:
        >>> llm = LiteLLM(model_path="models/tinyllama-1.1b-q4.gguf")
        >>> response = llm.generate("Hello, how are you?")
        >>> print(response)
    """

    # Recommended models for Pi 5 (download from HuggingFace)
    RECOMMENDED_MODELS = {
        "tinyllama": "TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf",
        "qwen2-0.5b": "qwen2-0.5b-instruct-q4_k_m.gguf",
        "phi3-mini": "Phi-3-mini-4k-instruct-q4.gguf",
        "gemma-2b": "gemma-2b-it-q4_k_m.gguf",
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = "tinyllama",
        n_ctx: int = 2048,
        n_threads: int = 4,
        n_gpu_layers: int = 0,
        verbose: bool = False,
    ):
        """Initialize LiteLLM.

        Args:
            model_path: Path to GGUF model file
            model_name: Preset model name if model_path not provided
            n_ctx: Context window size
            n_threads: Number of CPU threads (Pi 5 has 4 cores)
            n_gpu_layers: GPU layers (0 for Pi 5, no GPU)
            verbose: Whether to print llama.cpp logs
        """
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose

        self._llm = None
        self._model_path = model_path
        self._model_name = model_name

        # Conversation history
        self._messages = []
        self._system_prompt = None

    def _ensure_loaded(self):
        """Lazy-load the model on first use."""
        if self._llm is not None:
            return

        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed. Install with:\n"
                "pip install llama-cpp-python\n"
                "Or with OpenBLAS for better performance:\n"
                'CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python'
            )

        # Find model path
        if self._model_path:
            model_path = Path(self._model_path)
        else:
            # Look in models/ directory
            models_dir = Path("models")
            if self._model_name in self.RECOMMENDED_MODELS:
                model_path = models_dir / self.RECOMMENDED_MODELS[self._model_name]
            else:
                # Try to find any .gguf file
                gguf_files = list(models_dir.glob("*.gguf"))
                if gguf_files:
                    model_path = gguf_files[0]
                else:
                    raise FileNotFoundError(
                        f"No model found. Download a model to models/ directory:\n"
                        f"mkdir -p models && cd models\n"
                        f"wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
                    )

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"Loading LLM: {model_path.name}...")

        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_gpu_layers=self.n_gpu_layers,
            verbose=self.verbose,
        )

        print(f"LLM loaded ({self.n_threads} threads, {self.n_ctx} context)")

    def set_system_prompt(self, prompt: str):
        """Set the system prompt.

        Args:
            prompt: System prompt text
        """
        self._system_prompt = prompt
        self._messages = []  # Reset conversation

    def add_message(self, role: str, content: str):
        """Add a message to conversation history.

        Args:
            role: 'user' or 'assistant'
            content: Message content
        """
        self._messages.append({"role": role, "content": content})

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list] = None,
    ) -> str:
        """Generate a response.

        Args:
            prompt: User input
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            stop: Stop sequences

        Returns:
            Generated response text
        """
        self._ensure_loaded()

        # Build messages
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.extend(self._messages)
        messages.append({"role": "user", "content": prompt})

        # Generate
        response = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )

        # Extract response text
        assistant_message = response["choices"][0]["message"]["content"]

        # Update history
        self.add_message("user", prompt)
        self.add_message("assistant", assistant_message)

        return assistant_message

    def generate_streaming(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list] = None,
    ) -> Generator[str, None, None]:
        """Generate a response with streaming output.

        Args:
            prompt: User input
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            stop: Stop sequences

        Yields:
            Response text chunks
        """
        self._ensure_loaded()

        # Build messages
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.extend(self._messages)
        messages.append({"role": "user", "content": prompt})

        # Generate with streaming
        full_response = ""
        for chunk in self._llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            stream=True,
        ):
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                text = delta["content"]
                full_response += text
                yield text

        # Update history
        self.add_message("user", prompt)
        self.add_message("assistant", full_response)

    def reset(self):
        """Reset conversation history (keeps system prompt)."""
        self._messages = []


def create_lite_llm(
    model_path: Optional[str] = None,
    model_name: str = "tinyllama",
) -> LiteLLM:
    """Create a LiteLLM instance.

    Args:
        model_path: Path to GGUF model file
        model_name: Preset model name

    Returns:
        Configured LiteLLM instance
    """
    return LiteLLM(model_path=model_path, model_name=model_name)

"""Lightweight LLM using llama.cpp for Raspberry Pi CM4/Pi 5.

Uses llama-cpp-python bindings for efficient CPU inference.
Optimized for limited RAM (4GB) and no GPU acceleration.
"""

from pathlib import Path
from typing import Optional, Generator
import os
import urllib.request
import sys


def download_model(url: str, dest: Path, desc: str = "Downloading"):
    """Download a file with progress bar.

    Args:
        url: URL to download
        dest: Destination path
        desc: Description for progress bar
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    def progress_hook(count, block_size, total_size):
        percent = min(100, count * block_size * 100 // total_size)
        mb_done = count * block_size / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r{desc}: {percent}% ({mb_done:.1f}/{mb_total:.1f} MB)")
        sys.stdout.flush()

    print(f"{desc}...")
    urllib.request.urlretrieve(url, dest, progress_hook)
    print()  # Newline after progress


class LiteLLM:
    """Lightweight LLM wrapper using llama.cpp.

    Designed for Raspberry Pi 5 with small quantized models.

    Example:
        >>> llm = LiteLLM(model_path="models/tinyllama-1.1b-q4.gguf")
        >>> response = llm.generate("Hello, how are you?")
        >>> print(response)
    """

    # Recommended models for Pi CM4/Pi 5 with download URLs
    # Ordered by speed (fastest first)
    RECOMMENDED_MODELS = {
        # Fastest - 135M params, very quick responses (from QuantFactory - public)
        "smollm2-135m": {
            "file": "SmolLM2-135M-Instruct.Q8_0.gguf",
            "url": "https://huggingface.co/QuantFactory/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct.Q8_0.gguf",
            "size_mb": 145,
        },
        # Fast - 360M params, good balance
        "smollm2": {
            "file": "smollm2-360m-instruct-q8_0.gguf",
            "url": "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/resolve/main/smollm2-360m-instruct-q8_0.gguf",
            "size_mb": 386,
        },
        # Medium - 500M params
        "qwen2-0.5b": {
            "file": "qwen2-0_5b-instruct-q4_k_m.gguf",
            "url": "https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF/resolve/main/qwen2-0_5b-instruct-q4_k_m.gguf",
            "size_mb": 397,
        },
        # Slower - 1.1B params, best quality but slow on CM4
        "tinyllama": {
            "file": "TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf",
            "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "size_mb": 669,
        },
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = "smollm2",
        n_ctx: int = 512,
        n_threads: int = 3,
        n_batch: int = 64,
        n_gpu_layers: int = 0,
        use_mmap: bool = True,
        verbose: bool = False,
    ):
        """Initialize LiteLLM.

        Args:
            model_path: Path to GGUF model file
            model_name: Preset model name if model_path not provided
            n_ctx: Context window size (512 for speed, 1024 for longer conversations)
            n_threads: Number of CPU threads (3 optimal for CM4 thermals)
            n_batch: Batch size for prompt processing (64 good for CM4)
            n_gpu_layers: GPU layers (0 for CM4/Pi 5, no GPU)
            use_mmap: Memory-map model file (faster loading, less RAM)
            verbose: Whether to print llama.cpp logs
        """
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_batch = n_batch
        self.n_gpu_layers = n_gpu_layers
        self.use_mmap = use_mmap
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
        models_dir = Path("models")

        if self._model_path:
            model_path = Path(self._model_path)
        elif self._model_name in self.RECOMMENDED_MODELS:
            model_info = self.RECOMMENDED_MODELS[self._model_name]
            model_path = models_dir / model_info["file"]

            # Auto-download if not exists
            if not model_path.exists():
                print(f"Model '{self._model_name}' not found. Downloading ({model_info['size_mb']} MB)...")
                download_model(
                    url=model_info["url"],
                    dest=model_path,
                    desc=f"Downloading {self._model_name}",
                )
        else:
            # Try to find any .gguf file
            gguf_files = list(models_dir.glob("*.gguf")) if models_dir.exists() else []
            if gguf_files:
                model_path = gguf_files[0]
            else:
                # Default to tinyllama and download it
                model_info = self.RECOMMENDED_MODELS["tinyllama"]
                model_path = models_dir / model_info["file"]
                print(f"No model found. Downloading TinyLlama ({model_info['size_mb']} MB)...")
                download_model(
                    url=model_info["url"],
                    dest=model_path,
                    desc="Downloading tinyllama",
                    )

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"Loading LLM: {model_path.name}...")

        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_batch=self.n_batch,
            n_gpu_layers=self.n_gpu_layers,
            use_mmap=self.use_mmap,
            verbose=self.verbose,
        )

        print(f"LLM loaded ({self.n_threads} threads, {self.n_ctx} ctx, batch={self.n_batch})")

        # Try to get stop tokens from model metadata
        try:
            metadata = self._llm.metadata
            if metadata:
                # Look for EOS token and chat template info
                eos_token = metadata.get("tokenizer.ggml.eos_token_id")
                bos_token = metadata.get("tokenizer.ggml.bos_token_id")
                chat_template = metadata.get("tokenizer.chat_template")
                print(f"[Model] EOS: {eos_token}, BOS: {bos_token}")
                if chat_template:
                    print(f"[Model] Has chat template")
        except Exception as e:
            print(f"[Model] Could not read metadata: {e}")

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

    # Stop sequences by model family (prevent runaway generation)
    MODEL_STOP_TOKENS = {
        "tinyllama": ["</s>", "User:", "\nUser:", "Your capabilities"],
        "qwen": ["<|im_end|>", "<|endoftext|>", "User:", "\nUser:"],
        "smollm2-135m": ["<|im_end|>", "<|endoftext|>", "User:", "\nUser:"],
        "smollm": ["<|im_end|>", "<|endoftext|>", "User:", "\nUser:"],
        "default": ["User:", "\nUser:", "Human:", "\nHuman:", "Your capabilities"],
    }

    def _get_stop_sequences(self) -> list:
        """Get stop sequences appropriate for the loaded model."""
        model_name = (self._model_name or "").lower()
        for key in self.MODEL_STOP_TOKENS:
            if key in model_name:
                return self.MODEL_STOP_TOKENS[key]
        return self.MODEL_STOP_TOKENS["default"]

    def generate(
        self,
        prompt: str,
        max_tokens: int = 45,
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

        # Use model-specific stop sequences if none provided
        stop_seqs = stop if stop is not None else self._get_stop_sequences()

        # Generate
        response = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop_seqs,
        )

        # Extract response text
        assistant_message = response["choices"][0]["message"]["content"]

        # Clean up any leaked prompt patterns from response
        for stop_seq in stop_seqs:
            if stop_seq in assistant_message:
                assistant_message = assistant_message.split(stop_seq)[0]
        assistant_message = assistant_message.strip()

        # Update history
        self.add_message("user", prompt)
        self.add_message("assistant", assistant_message)

        return assistant_message

    def generate_streaming(
        self,
        prompt: str,
        max_tokens: int = 45,
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

        # Use model-specific stop sequences if none provided
        stop_seqs = stop if stop is not None else self._get_stop_sequences()

        # Generate with streaming
        full_response = ""
        for chunk in self._llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop_seqs,
            stream=True,
        ):
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                text = delta["content"]
                full_response += text
                yield text

        # Clean up any leaked prompt patterns from response
        for stop_seq in stop_seqs:
            if stop_seq in full_response:
                full_response = full_response.split(stop_seq)[0]

        # Update history
        self.add_message("user", prompt)
        self.add_message("assistant", full_response.strip())

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

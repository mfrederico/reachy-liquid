"""Conversation manager using LFM2-Audio model."""

import torch
import numpy as np

import config

# Import liquid_audio components
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState


class ConversationManager:
    """Manages conversation state and generates responses using LFM2-Audio."""

    def __init__(self, system_prompt: str = None):
        print(f"Loading LFM2-Audio model on {config.DEVICE}...")

        # Load model and processor
        self.processor = LFM2AudioProcessor.from_pretrained(
            config.LFM2_AUDIO_MODEL
        ).eval()

        self.model = LFM2AudioModel.from_pretrained(
            config.LFM2_AUDIO_MODEL,
            dtype=config.DTYPE,
            device=config.DEVICE,
        ).eval()

        # Initialize chat state
        self.chat = ChatState(self.processor)
        self._setup_system_prompt(system_prompt or config.SYSTEM_PROMPT)

        print("Model loaded successfully!")

    def _setup_system_prompt(self, prompt: str):
        """Set up the system prompt."""
        self.chat.new_turn("system")
        self.chat.add_text(prompt)
        self.chat.end_turn()

    def add_vision_context(self, detected_objects: list[dict]):
        """
        Add vision context to the conversation.

        Args:
            detected_objects: List of dicts with 'label', 'confidence', 'position'
        """
        if not detected_objects:
            return

        # Build description of what we see
        descriptions = []
        for obj in detected_objects:
            label = obj.get("label", "object")
            pos = obj.get("position", "")
            conf = obj.get("confidence", 0)
            if conf > config.DETECTION_CONFIDENCE:
                desc = f"{label}"
                if pos:
                    desc += f" ({pos})"
                descriptions.append(desc)

        if descriptions:
            context = f"[I can see: {', '.join(descriptions)}]"
            # This will be prepended to the user's audio context
            self._vision_context = context
        else:
            self._vision_context = None

    def generate_response_streaming(
        self,
        audio_input: np.ndarray,
        sample_rate: int = None,
    ):
        """
        Generate a response to audio input, yielding audio chunks as they're generated.

        Args:
            audio_input: Input audio as numpy array
            sample_rate: Sample rate of input audio (default: config.SAMPLE_RATE)

        Yields:
            torch.Tensor: Audio chunks (1920 samples each at 24kHz = 80ms)

        After iteration completes, access self.last_text_response for the text.
        """
        sample_rate = sample_rate or config.SAMPLE_RATE

        # Start new user turn
        self.chat.new_turn("user")

        # Add vision context if available
        if hasattr(self, "_vision_context") and self._vision_context:
            self.chat.add_text(self._vision_context)
            self._vision_context = None

        # Add audio input (must be 2D: [1, samples])
        audio_tensor = torch.from_numpy(audio_input).float()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        self.chat.add_audio(audio_tensor, sample_rate)
        self.chat.end_turn()

        # Generate response with streaming audio decode
        self.chat.new_turn("assistant")

        text_tokens = []

        # Use streaming context for proper audio decoding
        with torch.no_grad(), self.processor.mimi.streaming(1):
            for token in self.model.generate_interleaved(
                **self.chat,
                max_new_tokens=config.MAX_NEW_TOKENS,
                audio_temperature=config.AUDIO_TEMPERATURE,
                audio_top_k=config.AUDIO_TOP_K,
            ):
                if token.numel() == 1:
                    # Text token
                    decoded = self.processor.text.decode(token)
                    print(decoded, end="", flush=True)
                    text_tokens.append(token)
                elif token.numel() == 8:
                    # Audio token (8 codebooks) - decode and yield immediately
                    # Skip special token 2048
                    if (token == 2048).any():
                        continue
                    # Decode: shape [None, :, None] = [batch=1, codebooks=8, time=1]
                    wav_chunk = self.processor.mimi.decode(token[None, :, None])[0]
                    yield wav_chunk

        print()  # New line after text output

        # End assistant turn
        self.chat.end_turn()

        # Store text response for access after iteration
        if text_tokens:
            all_text_tokens = torch.cat(text_tokens)
            self.last_text_response = self.processor.text.decode(all_text_tokens)
        else:
            self.last_text_response = ""

    def reset(self):
        """Reset conversation history (keeps system prompt)."""
        self.chat = ChatState(self.processor)
        self._setup_system_prompt(config.SYSTEM_PROMPT)
        print("Conversation reset.")

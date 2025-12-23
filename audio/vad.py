"""Voice Activity Detection using Silero VAD."""

import torch
import numpy as np

import config


class VoiceActivityDetector:
    """Wrapper for Silero VAD model."""

    def __init__(self):
        self.model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        self.model.to(config.DEVICE)

        # Get utility functions
        (
            self.get_speech_timestamps,
            self.save_audio,
            self.read_audio,
            self.VADIterator,
            self.collect_chunks,
        ) = self.utils

        self.reset()

    def reset(self):
        """Reset VAD state for new utterance."""
        self.model.reset_states()

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        Check if audio chunk contains speech.

        Args:
            audio_chunk: Audio data as numpy array (mono, 16kHz)

        Returns:
            True if speech detected, False otherwise
        """
        # Convert to tensor
        if isinstance(audio_chunk, np.ndarray):
            audio_tensor = torch.from_numpy(audio_chunk).float()
        else:
            audio_tensor = audio_chunk.float()

        # Ensure correct shape (1D tensor)
        if audio_tensor.dim() > 1:
            audio_tensor = audio_tensor.squeeze()

        audio_tensor = audio_tensor.to(config.DEVICE)

        # Get speech probability
        with torch.no_grad():
            speech_prob = self.model(audio_tensor, config.SAMPLE_RATE).item()

        return speech_prob > config.VAD_THRESHOLD

    def get_speech_probability(self, audio_chunk: np.ndarray) -> float:
        """
        Get speech probability for audio chunk.

        Args:
            audio_chunk: Audio data as numpy array (mono, 16kHz)

        Returns:
            Speech probability (0.0 to 1.0)
        """
        if isinstance(audio_chunk, np.ndarray):
            audio_tensor = torch.from_numpy(audio_chunk).float()
        else:
            audio_tensor = audio_chunk.float()

        if audio_tensor.dim() > 1:
            audio_tensor = audio_tensor.squeeze()

        audio_tensor = audio_tensor.to(config.DEVICE)

        with torch.no_grad():
            return self.model(audio_tensor, config.SAMPLE_RATE).item()

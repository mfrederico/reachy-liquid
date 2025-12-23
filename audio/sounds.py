"""Simple sound effects for processing feedback.

Generates and plays short beeps/tones to indicate processing status.
"""

import numpy as np
import sounddevice as sd
import threading


class SoundEffects:
    """Play simple sound effects for processing feedback.

    Example:
        >>> sfx = SoundEffects()
        >>> sfx.processing()  # Play when starting to process
        >>> sfx.success()     # Play on successful completion
    """

    def __init__(self, sample_rate: int = 24000, volume: float = 0.3):
        self.sample_rate = sample_rate
        self.volume = volume

    def _generate_tone(
        self,
        frequency: float,
        duration: float,
        fade_ms: float = 10
    ) -> np.ndarray:
        """Generate a sine wave tone with fade in/out.

        Args:
            frequency: Tone frequency in Hz
            duration: Duration in seconds
            fade_ms: Fade in/out duration in milliseconds

        Returns:
            Audio samples as float32 array
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        tone = np.sin(2 * np.pi * frequency * t) * self.volume

        # Apply fade in/out to avoid clicks
        fade_samples = int(self.sample_rate * fade_ms / 1000)
        if fade_samples > 0 and len(tone) > fade_samples * 2:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            tone[:fade_samples] *= fade_in
            tone[-fade_samples:] *= fade_out

        return tone.astype(np.float32)

    def _generate_beep_sequence(
        self,
        frequencies: list[float],
        durations: list[float],
        gaps: list[float] = None
    ) -> np.ndarray:
        """Generate a sequence of beeps with gaps.

        Args:
            frequencies: List of frequencies for each beep
            durations: List of durations for each beep
            gaps: List of gaps between beeps (one less than frequencies)

        Returns:
            Combined audio samples
        """
        if gaps is None:
            gaps = [0.05] * (len(frequencies) - 1)

        parts = []
        for i, (freq, dur) in enumerate(zip(frequencies, durations)):
            parts.append(self._generate_tone(freq, dur))
            if i < len(gaps):
                gap_samples = int(self.sample_rate * gaps[i])
                parts.append(np.zeros(gap_samples, dtype=np.float32))

        return np.concatenate(parts)

    def _play_async(self, audio: np.ndarray):
        """Play audio in background thread."""
        def play():
            try:
                sd.play(audio, self.sample_rate, blocking=True)
            except Exception:
                pass  # Silently ignore audio errors

        thread = threading.Thread(target=play, daemon=True)
        thread.start()

    def processing(self):
        """Play a 'processing' sound - two quick ascending beeps."""
        audio = self._generate_beep_sequence(
            frequencies=[400, 600],
            durations=[0.08, 0.08],
            gaps=[0.03]
        )
        self._play_async(audio)

    def success(self):
        """Play a 'success' sound - cheerful ascending arpeggio."""
        audio = self._generate_beep_sequence(
            frequencies=[523, 659, 784],  # C5, E5, G5
            durations=[0.06, 0.06, 0.1],
            gaps=[0.02, 0.02]
        )
        self._play_async(audio)

    def error(self):
        """Play an 'error' sound - descending buzz."""
        audio = self._generate_beep_sequence(
            frequencies=[400, 300],
            durations=[0.1, 0.15],
            gaps=[0.02]
        )
        self._play_async(audio)

    def thinking(self):
        """Play a 'thinking' sound - single soft beep."""
        audio = self._generate_tone(440, 0.1)
        audio *= 0.5  # Quieter
        self._play_async(audio)


# Global instance for convenience
_sfx = None

def get_sounds() -> SoundEffects:
    """Get or create the global SoundEffects instance."""
    global _sfx
    if _sfx is None:
        _sfx = SoundEffects()
    return _sfx


def play_processing():
    """Play processing sound."""
    get_sounds().processing()

def play_success():
    """Play success sound."""
    get_sounds().success()

def play_error():
    """Play error sound."""
    get_sounds().error()

def play_thinking():
    """Play thinking sound."""
    get_sounds().thinking()

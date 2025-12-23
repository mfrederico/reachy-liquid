"""Audio playback for LFM2-Audio output."""

import queue
import threading
import numpy as np
import sounddevice as sd
import torch

import config


class AudioPlayer:
    """Plays audio from LFM2-Audio model output with streaming support."""

    def __init__(self):
        self.sample_rate = config.OUTPUT_SAMPLE_RATE
        self._stream = None
        self._queue = None
        self._finished = None

    def play(self, audio: np.ndarray | torch.Tensor, blocking: bool = True):
        """
        Play complete audio through speakers.

        Args:
            audio: Audio waveform (numpy array or torch tensor)
            blocking: If True, wait for playback to complete
        """
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        if audio.ndim > 1:
            audio = audio.squeeze()

        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val

        print(f"Playing response ({len(audio) / self.sample_rate:.2f}s)...")
        sd.play(audio, self.sample_rate)

        if blocking:
            sd.wait()
            print("Done.")

    def start_stream(self):
        """Start streaming audio playback."""
        self._queue = queue.Queue()
        self._finished = threading.Event()

        def callback(outdata, frames, time, status):
            try:
                data = self._queue.get_nowait()
                # Pad if chunk is smaller than requested frames
                if len(data) < frames:
                    outdata[:len(data), 0] = data
                    outdata[len(data):, 0] = 0
                else:
                    outdata[:, 0] = data[:frames]
            except queue.Empty:
                if self._finished.is_set():
                    raise sd.CallbackStop()
                outdata.fill(0)

        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=callback,
            blocksize=1920,  # Match mimi chunk size (80ms at 24kHz)
        )
        self._stream.start()

    def play_chunk(self, chunk: np.ndarray | torch.Tensor):
        """
        Queue an audio chunk for streaming playback.

        Args:
            chunk: Audio chunk (numpy array or torch tensor)
        """
        if self._queue is None:
            self.start_stream()

        if isinstance(chunk, torch.Tensor):
            chunk = chunk.cpu().numpy()

        if chunk.ndim > 1:
            chunk = chunk.squeeze()

        # Normalize chunk
        max_val = np.abs(chunk).max()
        if max_val > 1.0:
            chunk = chunk / max_val

        self._queue.put(chunk.astype(np.float32))

    def finish_stream(self):
        """Signal that no more chunks will be added and wait for playback to finish."""
        if self._finished is None:
            return

        self._finished.set()

        # Wait for queue to drain
        while not self._queue.empty():
            sd.sleep(50)

        # Small delay to ensure last chunk plays
        sd.sleep(100)

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        self._queue = None
        self._finished = None

    def stop(self):
        """Stop any currently playing audio."""
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        sd.stop()

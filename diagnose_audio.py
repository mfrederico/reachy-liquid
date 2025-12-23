#!/usr/bin/env python3
"""Diagnose audio issues - test recording and playback independently."""

import time
import numpy as np
import sounddevice as sd

def get_audio_devices():
    """List all audio devices."""
    print("=" * 60)
    print("AUDIO DEVICES")
    print("=" * 60)
    print(sd.query_devices())
    print()
    print(f"Default input:  {sd.default.device[0]}")
    print(f"Default output: {sd.default.device[1]}")
    print()

def test_recording(duration=2.0, sample_rate=16000):
    """Test recording without any processing."""
    print("=" * 60)
    print(f"RECORDING TEST ({duration}s at {sample_rate}Hz)")
    print("=" * 60)

    print("Recording... ", end="", flush=True)
    start = time.time()

    try:
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32,
            blocking=True,
        )
        elapsed = time.time() - start
        print(f"Done in {elapsed:.2f}s")

        # Check for issues
        max_val = np.abs(audio).max()
        mean_val = np.abs(audio).mean()
        print(f"  Max amplitude: {max_val:.4f}")
        print(f"  Mean amplitude: {mean_val:.4f}")

        if max_val < 0.01:
            print("  WARNING: Very low audio level - check microphone")

        return audio.flatten()

    except Exception as e:
        print(f"FAILED: {e}")
        return None

def test_playback_simple(duration=2.0, sample_rate=24000):
    """Test simple non-streaming playback."""
    print("=" * 60)
    print(f"SIMPLE PLAYBACK TEST ({duration}s sine wave)")
    print("=" * 60)

    # Generate test tone
    t = np.linspace(0, duration, int(duration * sample_rate))
    audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    print("Playing 440Hz tone... ", end="", flush=True)
    start = time.time()

    try:
        sd.play(audio, sample_rate)
        sd.wait()
        elapsed = time.time() - start
        print(f"Done in {elapsed:.2f}s")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False

def test_playback_streaming(duration=3.0, sample_rate=24000, chunk_ms=80):
    """Test streaming playback like we do with LFM2."""
    print("=" * 60)
    print(f"STREAMING PLAYBACK TEST ({duration}s, {chunk_ms}ms chunks)")
    print("=" * 60)

    chunk_samples = int(sample_rate * chunk_ms / 1000)
    total_samples = int(duration * sample_rate)

    # Generate test tone
    t = np.linspace(0, duration, total_samples)
    full_audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    # Split into chunks
    chunks = [full_audio[i:i+chunk_samples] for i in range(0, len(full_audio), chunk_samples)]

    print(f"Playing {len(chunks)} chunks of {chunk_samples} samples each...")

    import collections
    import threading

    buffer = collections.deque()
    buffer_lock = threading.Lock()
    finished = threading.Event()
    underruns = [0]

    def callback(outdata, frames, time_info, status):
        if status:
            print(f"  Status: {status}")

        with buffer_lock:
            available = len(buffer)
            if available >= frames:
                for i in range(frames):
                    outdata[i, 0] = buffer.popleft()
            elif available > 0:
                for i in range(available):
                    outdata[i, 0] = buffer.popleft()
                outdata[available:, 0] = 0
                underruns[0] += 1
            else:
                if finished.is_set():
                    raise sd.CallbackStop()
                outdata.fill(0)
                underruns[0] += 1

    # Prebuffer
    prebuffer_chunks = 5  # ~400ms
    for chunk in chunks[:prebuffer_chunks]:
        buffer.extend(chunk)

    stream = sd.OutputStream(
        samplerate=sample_rate,
        channels=1,
        callback=callback,
        blocksize=2048,
        latency='high',
    )

    start = time.time()
    stream.start()

    # Feed remaining chunks with simulated delay (like model generation)
    for i, chunk in enumerate(chunks[prebuffer_chunks:]):
        # Simulate model generation delay
        time.sleep(0.05)  # 50ms between chunks

        with buffer_lock:
            buffer.extend(chunk)

    finished.set()

    # Wait for playback
    while len(buffer) > 0:
        time.sleep(0.05)

    time.sleep(0.2)
    stream.stop()
    stream.close()

    elapsed = time.time() - start
    print(f"Done in {elapsed:.2f}s")
    print(f"  Buffer underruns: {underruns[0]}")

    if underruns[0] > 5:
        print("  WARNING: Many underruns - try larger prebuffer")

    return underruns[0]

def test_cpu_during_playback():
    """Test if CPU load affects playback."""
    print("=" * 60)
    print("CPU LOAD DURING PLAYBACK TEST")
    print("=" * 60)

    import threading

    # Generate audio
    duration = 3.0
    sample_rate = 24000
    t = np.linspace(0, duration, int(duration * sample_rate))
    audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    cpu_busy = threading.Event()

    def cpu_load():
        """Simulate CPU load like model inference."""
        while cpu_busy.is_set():
            # Busy loop
            _ = sum(range(100000))

    print("Playing without CPU load... ", end="", flush=True)
    sd.play(audio, sample_rate)
    sd.wait()
    print("Done")

    print("Playing WITH CPU load... ", end="", flush=True)
    cpu_busy.set()
    threads = [threading.Thread(target=cpu_load, daemon=True) for _ in range(4)]
    for t in threads:
        t.start()

    sd.play(audio, sample_rate)
    sd.wait()
    cpu_busy.clear()
    print("Done")

    print("  If second playback was choppy, CPU contention is the issue")

def check_system():
    """Check system audio configuration."""
    print("=" * 60)
    print("SYSTEM CHECK")
    print("=" * 60)

    import subprocess

    # Check PulseAudio/PipeWire
    try:
        result = subprocess.run(['pactl', 'info'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'Server Name' in line or 'Default Sink' in line:
                    print(f"  {line.strip()}")
    except Exception:
        print("  PulseAudio not available")

    # Check for realtime priority
    try:
        import os
        priority = os.nice(0)
        print(f"  Process nice level: {priority}")
    except Exception:
        pass

    print()

def main():
    print("\n" + "=" * 60)
    print("AUDIO DIAGNOSTICS")
    print("=" * 60 + "\n")

    check_system()
    get_audio_devices()

    input("Press Enter to test recording...")
    audio = test_recording()

    input("\nPress Enter to test simple playback...")
    test_playback_simple()

    input("\nPress Enter to test streaming playback...")
    underruns = test_playback_streaming()

    input("\nPress Enter to test CPU load impact...")
    test_cpu_during_playback()

    if audio is not None:
        input("\nPress Enter to play back your recording...")
        print("Playing your recording...")
        sd.play(audio, 16000)
        sd.wait()

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print("""
If streaming playback was choppy:
1. Try increasing prebuffer: AudioPlayer(prebuffer_ms=600)
2. Try 'high' latency mode (already set)
3. Check if PulseAudio is adding latency

If CPU load test was choppy:
1. Model inference is blocking audio - need process isolation
2. Try running audio in separate process (multiprocessing)

If recording was choppy:
1. Check microphone settings
2. Try different sample rate
""")

if __name__ == "__main__":
    main()

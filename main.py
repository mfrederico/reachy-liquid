#!/usr/bin/env python3
"""
Liquid Reachy - Conversational AI Robot Companion

A voice-controlled AI assistant with vision capabilities.
Uses LFM2-Audio for speech-to-speech conversation and YOLOv8 for object detection.
"""

import argparse
import signal
import sys

import config


def main():
    parser = argparse.ArgumentParser(
        description="Liquid Reachy - Conversational AI Companion"
    )
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="Disable vision/camera",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Run in text-only mode (for testing)",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("  Liquid Reachy - AI Companion")
    print("=" * 50)
    print(f"Device: {config.DEVICE}")
    print()

    # Import components (deferred to show startup progress)
    from audio import AudioRecorder, AudioPlayer
    from conversation import ConversationManager

    # Initialize components
    print("Initializing components...")

    # Audio
    recorder = AudioRecorder()
    player = AudioPlayer()

    # Conversation (loads model)
    conversation = ConversationManager()

    # Vision (optional)
    camera = None
    detector = None

    if not args.no_vision:
        try:
            from vision import Camera, ObjectDetector

            camera = Camera()
            if camera.start():
                detector = ObjectDetector()
            else:
                print("Vision disabled (camera failed to start)")
                camera = None
        except Exception as e:
            print(f"Vision disabled: {e}")
            camera = None

    print()
    print("=" * 50)
    print("  Ready! Speak to start a conversation.")
    print("  Press Ctrl+C to exit.")
    print("=" * 50)
    print()

    # Handle graceful shutdown
    running = True

    def signal_handler(sig, frame):
        nonlocal running
        print("\nShutting down...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Main conversation loop
    audio_input = None

    try:
        while running:
            # Get vision context if available
            if camera and detector:
                frame = camera.get_frame()
                if frame is not None:
                    detections = detector.detect(frame)
                    conversation.add_vision_context(detections)

            # Wait for and record speech (unless we have barge-in audio)
            if audio_input is None:
                audio_input = recorder.record_speech(timeout=None)

            if audio_input is None:
                continue

            if not running:
                break

            # Generate and stream response with barge-in detection
            print("\nAssistant: ", end="")
            player.start_stream()
            recorder.start_barge_in_monitor(delay=0.5)  # Wait 500ms before monitoring

            chunk_count = 0
            interrupted = False

            for audio_chunk in conversation.generate_response_streaming(audio_input):
                player.play_chunk(audio_chunk)
                chunk_count += 1

                # Check for barge-in (non-threaded, checks between chunks)
                if recorder.check_barge_in():
                    print("\n[Interrupted]")
                    interrupted = True
                    break

            # Stop monitoring
            barge_in = recorder.stop_barge_in_monitor()

            if interrupted or barge_in:
                # User interrupted - stop playback and record their speech
                player.stop()
                audio_input = recorder.record_after_barge_in()
                print()  # New line after interrupted response
            else:
                # Normal completion - let audio finish playing
                player.finish_stream()
                audio_input = None  # Will wait for new speech

                if chunk_count == 0:
                    print("(No audio response generated)")

                print()

    except Exception as e:
        print(f"\nError: {e}")
        raise

    finally:
        # Cleanup
        recorder.stop_barge_in_monitor()
        player.stop()
        if camera:
            camera.stop()
        print("Goodbye!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Liquid Reachy - Conversational AI Robot Companion

A voice-controlled AI assistant with vision capabilities.
Uses LFM2-Audio for speech-to-speech conversation and YOLOv8 for object detection.
Optionally uses LFM2-Tool or keyword matching for camera PTZ control.
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
        "--lite",
        action="store_true",
        help="Use lightweight stack for Pi 5: Whisper + LLM + Piper TTS (no LFM2-Audio)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tinyllama",
        choices=["tinyllama", "qwen2-0.5b", "smollm2"],
        help="LLM model for --lite mode (default: tinyllama). Auto-downloads if not present.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Custom path to GGUF model file for --lite mode",
    )
    parser.add_argument(
        "--tts-voice",
        type=str,
        default="amy",
        choices=["amy", "lessac", "ryan", "arctic", "jenny", "alan"],
        help="Piper TTS voice for --lite mode (default: amy)",
    )
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="Disable vision/camera",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=config.VOICE_PRESET,
        choices=list(config.VOICE_PRESETS.keys()),
        help=f"Voice preset for LFM2-Audio (default: {config.VOICE_PRESET})",
    )
    parser.add_argument(
        "--tools",
        type=str,
        nargs="?",
        const="keywords",
        choices=["keywords", "llm"],
        help="Enable camera control: 'keywords' (Pi-friendly, needs whisper) or 'llm' (LFM2-Tool, needs GPU)",
    )
    parser.add_argument(
        "--transcribe",
        action="store_true",
        help="Show transcription of what you said (uses Whisper)",
    )
    parser.add_argument(
        "--whisper",
        type=str,
        default="tiny",
        choices=["tiny", "base", "small"],
        help="Whisper model size for transcription (default: tiny)",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Show system resource usage",
    )
    parser.add_argument(
        "--prebuffer",
        type=float,
        default=0.8,
        help="Audio prebuffer in seconds (default: 0.8, increase if jittery)",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("  Liquid Reachy - AI Companion")
    print("=" * 50)

    if args.lite:
        print("Mode: LITE (Pi 5 compatible)")
        print(f"LLM: {args.model}")
        print(f"TTS: Piper ({args.tts_voice})")
        print(f"ASR: Whisper {args.whisper}")
    else:
        print(f"Device: {config.DEVICE}")
        print(f"Voice: {args.voice}")
        if args.tools == "keywords":
            print(f"Tools: Keywords (Whisper {args.whisper})")
        elif args.tools == "llm":
            print("Tools: LFM2-Tool (LLM-based)")
        if args.transcribe:
            print(f"Transcription: Whisper {args.whisper}")

    print(f"Audio prebuffer: {args.prebuffer}s")
    print()

    # Show initial resource usage if monitoring
    monitor = None
    if args.monitor:
        from monitor import ResourceMonitor
        monitor = ResourceMonitor()
        print(f"Initial: {monitor.get_stats().short_str()}")
        print()

    # Import components (deferred to show startup progress)
    from audio import AudioRecorder, AudioPlayer

    # Initialize components
    print("Initializing components...")

    # Audio I/O
    recorder = AudioRecorder()
    player = AudioPlayer(prebuffer_seconds=args.prebuffer)

    # Transcriber (for showing what user said, and/or keyword tools)
    # Lite mode needs transcription for both display and keyword tools
    transcriber = None
    if args.transcribe or args.tools == "keywords" or args.lite:
        from audio import create_transcriber
        transcriber = create_transcriber(
            enabled=True,
            model_size=args.whisper,
            device="cpu" if args.lite else config.DEVICE,  # Lite mode uses CPU
        )

    # Vision (optional) - initialize before conversation so we can pass camera to tools
    camera = None
    detector = None

    if not args.no_vision:
        try:
            from vision import PTZCamera, ObjectDetector

            camera = PTZCamera()
            if camera.start():
                detector = ObjectDetector()
                if camera.has_ptz:
                    print("PTZ controls available")
            else:
                print("Vision disabled (camera failed to start)")
                camera = None
        except Exception as e:
            print(f"Vision disabled: {e}")
            camera = None

    # Conversation manager
    keyword_matcher = None
    has_ptz = camera and camera.has_ptz
    has_tools = args.tools == "keywords"

    # Generate system prompt with capability awareness
    system_prompt = config.get_system_prompt(
        voice_preset=args.voice if not args.lite else "neutral",
        has_camera=has_ptz,
        has_tools=has_tools or args.lite,  # Lite mode always has tools
    )

    if args.lite:
        # Lightweight Pi 5 stack: Whisper + LLM + Piper TTS
        from conversation import LiteConversationManager
        conversation = LiteConversationManager(
            system_prompt=system_prompt,
            model_path=args.model_path,
            model_name=args.model,
            voice=args.tts_voice,
            whisper_model=args.whisper,
        )

        # Always enable keyword tools in lite mode
        from tools import KeywordToolMatcher
        keyword_matcher = KeywordToolMatcher(camera=camera, detector=detector)
        print("Keyword tools enabled (time, date" + (", PTZ" if has_ptz else "") + ")")

    elif args.tools == "llm":
        # Full LLM-based tool calling (needs GPU + VRAM)
        from conversation import HybridModelManager
        conversation = HybridModelManager(
            system_prompt=system_prompt,
            camera=camera,
            detector=detector,
            load_tool_model=True,
        )
    else:
        # Basic conversation (optionally with keyword-based tools)
        from conversation import ConversationManager
        conversation = ConversationManager(system_prompt=system_prompt)

        if args.tools == "keywords":
            from tools import KeywordToolMatcher
            keyword_matcher = KeywordToolMatcher(camera=camera, detector=detector)
            print("Keyword tools enabled (time, date" + (", PTZ" if has_ptz else "") + ")")

    # Show resource usage after loading models
    if monitor:
        print()
        print(f"After loading: {monitor.get_stats().short_str()}")

    print()
    print("=" * 50)
    print("  Ready! Speak to start a conversation.")
    if args.tools == "keywords":
        print("  Commands: 'what time is it', 'what's the date'")
        if camera and camera.has_ptz:
            print("  Camera: 'look left/right/up/down'")
    print("  Press Ctrl+C to exit.")
    print("=" * 50)
    print()

    # Handle graceful shutdown
    running = True
    shutdown_count = 0

    def signal_handler(sig, frame):
        nonlocal running, shutdown_count
        shutdown_count += 1

        if shutdown_count == 1:
            print("\nShutting down...")
            running = False
            recorder.stop()  # Signal recorder to stop blocking
            player.stop()    # Stop any playing audio
        elif shutdown_count >= 2:
            print("\nForce exit!")
            import os
            os._exit(1)

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

            # Transcribe user speech (if enabled)
            user_text = None
            if transcriber:
                user_text = transcriber.transcribe(audio_input, sample_rate=config.SAMPLE_RATE)
                if user_text:
                    print(f"\nYou: {user_text}")

            # Check for commands via keywords (if enabled)
            tool_results = []
            if keyword_matcher and user_text:
                tool_results = keyword_matcher.process(user_text)
                for result in tool_results:
                    if result.get("success"):
                        action = result.get("action", "done")
                        print(f"[Tool: {action}]")

            # Pass tool results to conversation so LLM knows about them
            # Use pre_speak=True for keywords/lite (guaranteed output), False for LLM tools
            if tool_results:
                use_pre_speak = (args.tools == "keywords") or args.lite
                conversation.add_tool_context(tool_results, pre_speak=use_pre_speak)

            # Generate and stream response with barge-in detection
            print("\nAssistant: ", end="")
            player.start_stream()
            recorder.start_barge_in_monitor(delay=0.5)

            chunk_count = 0
            interrupted = False

            for audio_chunk in conversation.generate_response_streaming(audio_input, user_text=user_text):
                player.play_chunk(audio_chunk)
                chunk_count += 1

                # Check for barge-in
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
                print()
            else:
                # Normal completion - let audio finish playing
                player.finish_stream()
                audio_input = None

                if chunk_count == 0:
                    print("(No audio response generated)")

                print()

            # Show resource usage if monitoring
            if monitor and chunk_count > 0:
                print(f"[{monitor.get_stats().short_str()}]")

    except Exception as e:
        print(f"\nError: {e}")
        raise

    finally:
        # Cleanup
        recorder.stop_barge_in_monitor()
        recorder.close()  # Close persistent input stream
        player.close()    # Close persistent output stream
        if camera:
            camera.stop()
        print("Goodbye!")


if __name__ == "__main__":
    main()

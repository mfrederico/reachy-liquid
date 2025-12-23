"""Hybrid model manager for LFM2-Audio + LFM2-Tool.

Runs both models in parallel:
- LFM2-Audio: Speech-to-speech conversation
- LFM2-Tool: Text-based tool calling for camera control, etc.

The flow:
1. User speaks → Audio recorded
2. LFM2-Audio processes audio, outputs text + audio
3. Extracted text is also sent to LFM2-Tool for potential tool calls
4. If tools are called, results are injected into the conversation
5. Audio response is streamed to user
"""

import json
from typing import Generator

import torch
import numpy as np

import config
from tools import PTZ_TOOLS, ToolExecutor
# Sound effects disabled - conflicts with persistent audio streams
# from audio import play_processing, play_success


class HybridModelManager:
    """Manages LFM2-Audio and LFM2-Tool models together.

    LFM2-Audio handles the voice conversation.
    LFM2-Tool analyzes text to decide when to call camera/vision tools.

    Example:
        >>> manager = HybridModelManager(camera=ptz_camera, detector=detector)
        >>> for audio_chunk, text in manager.process_speech(audio_input):
        ...     player.play_chunk(audio_chunk)
        ...     if text:
        ...         print(text, end="")
    """

    def __init__(
        self,
        system_prompt: str = None,
        camera=None,
        detector=None,
        load_tool_model: bool = True,
    ):
        """Initialize hybrid model manager.

        Args:
            system_prompt: System prompt for LFM2-Audio
            camera: PTZCamera instance for vision/PTZ tools
            detector: ObjectDetector instance for vision
            load_tool_model: Whether to load LFM2-Tool (set False to save VRAM)
        """
        self.system_prompt = system_prompt or config.SYSTEM_PROMPT
        self.camera = camera
        self.detector = detector

        # Tool executor
        self.tool_executor = ToolExecutor(camera=camera, detector=detector)

        # Models
        self.audio_model = None
        self.audio_processor = None
        self.audio_chat = None

        self.tool_model = None
        self.tool_tokenizer = None

        self._load_tool_model = load_tool_model
        self._vision_context = []

        # Load models
        self._load_models()

    def _load_models(self):
        """Load both models."""
        print("Loading LFM2-Audio model...")
        self._load_audio_model()

        if self._load_tool_model:
            print("Loading LFM2-Tool model...")
            self._load_tool_model_impl()

    def _load_audio_model(self):
        """Load LFM2-Audio for speech-to-speech."""
        from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState

        self.audio_processor = LFM2AudioProcessor.from_pretrained(
            config.LFM2_AUDIO_MODEL
        ).eval()

        self.audio_model = LFM2AudioModel.from_pretrained(
            config.LFM2_AUDIO_MODEL,
            dtype=config.DTYPE,
            device=config.DEVICE,
        ).eval()

        # Initialize chat state with system prompt
        self.audio_chat = ChatState(self.audio_processor)
        self.audio_chat.new_turn("system")
        self.audio_chat.add_text(self.system_prompt)
        self.audio_chat.end_turn()
        print(f"LFM2-Audio loaded on {config.DEVICE}")

    def _load_tool_model_impl(self):
        """Load LFM2-Tool for function calling."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "LiquidAI/LFM2-1.2B-Tool"

        self.tool_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tool_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=config.DTYPE,
            device_map=config.DEVICE,
        )
        self.tool_model.eval()
        print(f"LFM2-Tool loaded on {config.DEVICE}")

    def _get_tool_system_prompt(self) -> str:
        """Build system prompt with tool definitions for LFM2-Tool."""
        tools_json = json.dumps(PTZ_TOOLS)
        return f"""You are a helpful assistant with camera control capabilities.
When the user asks you to look at something, find something, or adjust the view, use the appropriate tool.

List of tools: <|tool_list_start|>{tools_json}<|tool_list_end|>

Only call tools when the user explicitly asks to look somewhere, find something, or adjust the camera.
For normal conversation, just respond normally without calling tools."""

    def check_for_tool_calls(self, user_text: str) -> list[tuple[str, dict]]:
        """Check if user input should trigger any tool calls.

        Uses LFM2-Tool to analyze the text and decide on tools.

        Args:
            user_text: Transcribed user speech or text input

        Returns:
            List of (function_name, arguments) tuples
        """
        if self.tool_model is None:
            return []

        # Phrase-based filter to avoid false positives (e.g., "mountain times on" → not a time query)
        text_lower = user_text.lower()

        # Camera/PTZ phrases - require action words
        camera_phrases = [
            "look left", "look right", "look up", "look down",
            "turn left", "turn right", "pan left", "pan right",
            "tilt up", "tilt down", "zoom in", "zoom out",
            "look around", "look at", "look for", "look toward",
            "what do you see", "what can you see", "describe",
            "center camera", "reset camera",
        ]

        # Time/date phrases - require question context
        time_phrases = [
            "what time", "current time", "time is it", "time now",
            "what's the time", "tell me the time", "check the time",
            "time in ", "time zone",  # "time in bali", "time zone"
            "what date", "what day", "today's date", "current date",
            "what's the date", "tell me the date",
        ]

        all_phrases = camera_phrases + time_phrases
        if not any(phrase in text_lower for phrase in all_phrases):
            return []

        try:
            messages = [
                {"role": "system", "content": self._get_tool_system_prompt()},
                {"role": "user", "content": user_text}
            ]

            input_text = self.tool_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tool_tokenizer(input_text, return_tensors="pt")
            inputs = {k: v.to(config.DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.tool_model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=self.tool_tokenizer.eos_token_id,
                )

            response = self.tool_tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=False
            )

            # Extract tool calls
            return self.tool_executor.extract_tool_calls(response)

        except Exception as e:
            print(f"Tool check error: {e}")
            return []

    def execute_tools(self, calls: list[tuple[str, dict]]) -> list[dict]:
        """Execute tool calls and return results.

        Args:
            calls: List of (function_name, arguments) tuples

        Returns:
            List of result dictionaries
        """
        results = []
        for func_name, args in calls:
            print(f"[Tool] {func_name}({args})")
            result = self.tool_executor.execute(func_name, args)
            results.append(result)
            print(f"[Tool] Result: {result}")
        return results

    def add_vision_context(self, detections: list[dict]):
        """Add detected objects as context for the conversation.

        Args:
            detections: List of detection dicts from ObjectDetector
        """
        self._vision_context = detections

    def add_tool_context(self, tool_results: list[dict], pre_speak: bool = False):
        """Add tool execution results to the conversation context.

        For HybridModelManager, tool results are handled via LFM2-Tool's native
        tool calling, so this just stores them for potential injection.

        Args:
            tool_results: List of tool result dicts
            pre_speak: Ignored for LLM mode (model handles naturally)
        """
        # Store for potential use - LFM2-Tool handles this more natively
        self._pending_tool_results = tool_results

    def _format_vision_context(self) -> str:
        """Format vision context as text for the model."""
        if not self._vision_context:
            return ""

        objects = [d.get("name", "object") for d in self._vision_context[:5]]
        return f"[Currently visible: {', '.join(objects)}]"

    def _format_tool_results(self, results: list[dict]) -> str:
        """Format tool results as context for the model."""
        if not results:
            return ""

        summaries = []
        for r in results:
            if r.get("success"):
                # Prefer 'message' for information tools (time, date, describe)
                # Fall back to 'action' for action tools (look_left, zoom_in)
                if "message" in r:
                    summaries.append(r["message"])
                else:
                    action = r.get("action", "completed action")
                    summaries.append(action)
                    if "objects" in r:
                        obj_names = [o["name"] for o in r["objects"]]
                        if obj_names:
                            summaries.append(f"I can see: {', '.join(obj_names)}")
            else:
                summaries.append(f"couldn't do that: {r.get('error', 'unknown error')}")

        return "; ".join(summaries)

    def generate_response_streaming(
        self,
        audio_input: np.ndarray,
        sample_rate: int = None,
        user_text: str = None,
    ) -> Generator[np.ndarray, None, None]:
        """Process speech input and generate streaming audio response.

        New flow (tool-first):
        1. Use transcribed text (user_text) to check for tool calls via LFM2-Tool
        2. Execute any tools BEFORE generating response
        3. Inject tool results as context
        4. Generate speech response with LFM2-Audio

        Args:
            audio_input: Audio waveform as numpy array
            sample_rate: Sample rate (default: config.SAMPLE_RATE)
            user_text: Pre-transcribed text from Whisper (for tool checking)

        Yields:
            Audio chunks as numpy arrays
        """
        sample_rate = sample_rate or config.SAMPLE_RATE

        # Step 1: Quick keyword check - only invoke LFM2-Tool if looks like tool request
        tool_results = []
        tool_context = ""

        if user_text and self.tool_model is not None:
            # Phrase-based filter to reduce false positives
            text_lower = user_text.lower()

            # Camera/PTZ phrases - require action words
            camera_phrases = [
                "look left", "look right", "look up", "look down",
                "turn left", "turn right", "pan left", "pan right",
                "tilt up", "tilt down", "zoom in", "zoom out",
                "look around", "look at", "look for", "look toward",
                "what do you see", "what can you see", "describe",
                "center camera", "reset camera",
            ]

            # Time/date phrases - require question context
            time_phrases = [
                "what time", "current time", "time is it", "time now",
                "what's the time", "tell me the time", "check the time",
                "time in ", "time zone",  # "time in bali", "time zone"
                "what date", "what day", "today's date", "current date",
                "what's the date", "tell me the date",
            ]

            all_phrases = camera_phrases + time_phrases
            might_be_tool_request = any(phrase in text_lower for phrase in all_phrases)

            if might_be_tool_request:
                # Only now invoke the expensive LFM2-Tool check
                print("[Checking tools...]", end=" ", flush=True)
                tool_calls = self.check_for_tool_calls(user_text)
                if tool_calls:
                    tool_results = self.execute_tools(tool_calls)
                    tool_context = self._format_tool_results(tool_results)
                else:
                    print("[No tools needed]")

        # Convert audio to tensor
        if isinstance(audio_input, np.ndarray):
            audio_tensor = torch.from_numpy(audio_input).float()
        else:
            audio_tensor = audio_input

        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        # Resample if needed
        if sample_rate != config.SAMPLE_RATE:
            import torchaudio
            audio_tensor = torchaudio.functional.resample(
                audio_tensor, sample_rate, config.SAMPLE_RATE
            )

        # Step 2: Build user turn with all context
        self.audio_chat.new_turn("user")

        # Add vision context if available
        vision_text = self._format_vision_context()
        if vision_text:
            self.audio_chat.add_text(vision_text)

        # Pre-speak tool results - LFM2-Audio doesn't reliably incorporate injected context
        if tool_context:
            # Print the factual information so user sees correct data
            print(f"[{tool_context}] ", end="", flush=True)
            # Still inject as context, though model may ignore it
            self.audio_chat.add_text(f"SYSTEM: {tool_context}")

        # Add the audio
        self.audio_chat.add_audio(audio_tensor, config.SAMPLE_RATE)
        self.audio_chat.end_turn()

        # Step 3: Generate response with LFM2-Audio
        self.audio_chat.new_turn("assistant")

        collected_text = []

        with torch.no_grad(), self.audio_processor.mimi.streaming(1):
            for token in self.audio_model.generate_interleaved(
                **self.audio_chat,
                max_new_tokens=config.MAX_NEW_TOKENS,
                audio_temperature=config.AUDIO_TEMPERATURE,
                audio_top_k=config.AUDIO_TOP_K,
            ):
                if token.numel() == 1:
                    # Text token
                    decoded = self.audio_processor.text.decode(token)
                    collected_text.append(decoded)
                    print(decoded, end="", flush=True)

                elif token.numel() == 8:
                    # Audio token - decode and yield
                    if (token == 2048).any():
                        continue
                    wav_chunk = self.audio_processor.mimi.decode(token[None, :, None])[0]
                    yield wav_chunk

        # End assistant turn
        self.audio_chat.end_turn()

        # Store text response
        self.last_text_response = "".join(collected_text)

        print()  # Newline after response

    def reset_conversation(self):
        """Reset the conversation state."""
        from liquid_audio import ChatState
        self.audio_chat = ChatState(self.audio_processor)
        self.audio_chat.new_turn("system")
        self.audio_chat.add_text(self.system_prompt)
        self.audio_chat.end_turn()
        self._vision_context = []


# Convenience function to create manager with monitoring
def create_hybrid_manager(
    system_prompt: str = None,
    camera=None,
    detector=None,
    enable_tools: bool = True,
    show_memory: bool = True,
) -> HybridModelManager:
    """Create a HybridModelManager with optional resource monitoring.

    Args:
        system_prompt: System prompt for the conversation
        camera: PTZCamera instance
        detector: ObjectDetector instance
        enable_tools: Whether to load LFM2-Tool
        show_memory: Whether to show memory usage after loading

    Returns:
        Configured HybridModelManager
    """
    manager = HybridModelManager(
        system_prompt=system_prompt,
        camera=camera,
        detector=detector,
        load_tool_model=enable_tools,
    )

    if show_memory:
        from monitor import get_system_stats
        stats = get_system_stats()
        print(f"\nResource usage: {stats.short_str()}")

    return manager

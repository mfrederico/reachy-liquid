# Tool Calling with LFM2-Audio

LFM2-Audio is a speech-to-speech model that doesn't natively support tool calling. This document outlines approaches to enable tool functionality.

## Current Implementation

### Keyword Matching (`--tools keywords`)
- Uses Whisper to transcribe user speech
- Regex patterns match commands like "look left", "what time is it"
- Lightweight, works on Raspberry Pi
- No additional GPU memory required
- Limited to predefined patterns

### LLM-Based (`--tools llm`)
- Runs LFM2-Tool (1.2B) alongside LFM2-Audio
- LFM2-Tool analyzes text and decides when to call tools
- More flexible than keywords
- Requires additional ~2GB VRAM

## Limitations

Both approaches inject tool results as text context, but LFM2-Audio wasn't trained to incorporate injected context reliably. The model may:
- Ignore the injected information
- Claim it can't do things it actually did (e.g., "I can't move the camera" after moving it)
- Hallucinate information instead of using provided data

## Future Approaches

### 1. LoRA Fine-tuning (Recommended)
Train a small adapter (~1-5% of model parameters) on tool-calling data.

**Pros:**
- Can be done on a single GPU in a few hours
- Minimal model modification
- Preserves base model capabilities

**Cons:**
- Requires curated training dataset
- May need periodic retraining

**Dataset format example:**
```json
[
  {
    "audio": "look_to_the_left.wav",
    "transcript": "Can you look to the left?",
    "response": "<tool>pan_left</tool> Looking left now.",
    "response_audio": "looking_left_now.wav"
  },
  {
    "audio": "what_time_is_it.wav",
    "transcript": "What time is it?",
    "response": "<tool>get_time</tool> The current time is {time}.",
    "response_audio": null
  }
]
```

**Training considerations:**
- Need ~1000+ examples for reliable tool calling
- Include both tool-calling and non-tool conversations
- Audio pairs for speech-to-speech training

### 2. Special Token Integration
Add tool-specific tokens to the vocabulary:
- `<|tool_call_start|>`
- `<|tool_name|>`
- `<|tool_args|>`
- `<|tool_call_end|>`
- `<|tool_result|>`

Similar to how LFM2-Tool works, but integrated into the audio model.

**Pros:**
- Native tool calling support
- Clean separation of text/tool/audio tokens

**Cons:**
- Requires significant retraining
- May need Liquid AI's involvement for proper implementation

### 3. Output Parsing
Train or prompt the model to output structured text that can be parsed:

```
USER: Look to the left please
ASSISTANT: ACTION:pan_left RESPONSE:I'll look left for you.
```

**Pros:**
- Works with existing model architecture
- Can be implemented with prompt engineering

**Cons:**
- Fragile, model may not consistently follow format
- Adds latency for parsing

### 4. Parallel Analysis Architecture
Current hybrid approach, but improved:
1. Audio input → LFM2-Audio (conversation) + Small classifier (tool detection)
2. Classifier runs in parallel, decides if tool needed
3. Tool results injected before audio generation starts

**Pros:**
- Minimal latency impact
- Can use tiny specialized models for tool detection

**Cons:**
- Adds complexity
- Still relies on context injection

## Recommended Path Forward

1. **Short term:** Continue with keyword matching for reliability
2. **Medium term:** Collect tool-calling conversation data
3. **Long term:** LoRA fine-tune for native tool support

## Resources

- [LFM2-Audio Model](https://huggingface.co/LiquidAI/LFM2-Audio-1.5B)
- [LFM2-Tool Model](https://huggingface.co/LiquidAI/LFM2-1.2B-Tool)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Library](https://github.com/huggingface/peft) - For LoRA training

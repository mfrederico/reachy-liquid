# Liquid Reachy

A lightweight conversational AI companion for Reachy Mini. Uses a pipeline of Whisper (ASR) + Small LLM + Piper TTS for voice conversations.

## Setup

```bash
# Activate the virtual environment
source /venvs/mini_daemon/bin/activate

# Install dependencies
uv pip install -r requirements-lite.txt
```

### Note on llama-cpp-python

For better performance, install with OpenBLAS:

```bash
CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" uv pip install llama-cpp-python
```

## Usage

```bash
python main.py --lite
```

The LLM model will auto-download on first run (~150-400MB depending on model choice).

## Command Line Options

### Core Options

| Option | Default | Description |
|--------|---------|-------------|
| `--lite` | - | Use lightweight stack (required for Reachy Mini) |
| `--model` | `smollm2` | LLM model: `smollm2-135m`, `smollm2`, `qwen2-0.5b`, `tinyllama` |
| `--model-path` | - | Custom path to GGUF model file |
| `--tts-voice` | `amy` | Piper voice: `amy`, `lessac`, `ryan`, `arctic`, `jenny`, `alan` (add `-fast` for lower latency) |
| `--whisper` | `tiny` | Whisper model size: `tiny`, `base`, `small` |

### Additional Options

| Option | Default | Description |
|--------|---------|-------------|
| `--no-vision` | - | Disable camera/vision |
| `--monitor` | - | Show system resource usage |
| `--prebuffer` | `0.8` | Audio prebuffer in seconds (increase if audio is jittery) |

## Examples

```bash
# Fastest response time (smallest models)
python main.py --lite --model smollm2-135m --tts-voice amy-fast --whisper tiny

# Better quality responses (larger LLM)
python main.py --lite --model tinyllama --tts-voice amy

# No camera, with resource monitoring
python main.py --lite --no-vision --monitor
```

## Models

| Model | Parameters | Size | Speed |
|-------|------------|------|-------|
| `smollm2-135m` | 135M | 145 MB | Fastest |
| `smollm2` | 360M | 386 MB | Fast |
| `qwen2-0.5b` | 500M | 397 MB | Medium |
| `tinyllama` | 1.1B | 669 MB | Slowest |

Models are downloaded automatically to the `models/` directory on first use.

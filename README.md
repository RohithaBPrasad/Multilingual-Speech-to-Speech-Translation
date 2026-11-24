# Multilingual Speech-to-Speech Translation

A Python speech-to-speech translation system built with a cascaded machine-learning pipeline:

1. **Automatic speech recognition (ASR):** Faster-Whisper (`small`).
2. **Machine translation (MT):** Meta NLLB-200 (`facebook/nllb-200-distilled-600M`).
3. **Text-to-speech (TTS):** Meta MMS language models using VITS.

The project supports an HTTP API, a WebSocket audio bridge, real-time microphone translation,
and local audio-file testing.

## Project structure

```text
Multilingual_System/
├── pipeline.py       # ASR, translation, TTS and orchestration
├── server.py         # FastAPI HTTP and WebSocket service
├── realtime_mic.py   # Continuous microphone translation with Silero VAD
├── test_local.py    # Translate a local WAV file
├── test_mic.py      # Record and translate a microphone sample
├── requirements.txt # Python dependencies
├── pyproject.toml    # Formatter and linter configuration
└── .gitignore
```

## Requirements

- Python 3.10 or newer
- A working audio input/output device for microphone mode
- FFmpeg may be required by the local Faster-Whisper/CTranslate2 installation, depending on platform
- Internet access on first run to download the Hugging Face model files
- A CUDA-compatible PyTorch installation is optional; CPU execution is supported but slower

> Model weights are downloaded by the libraries at runtime. They are not stored in this repository.

## Installation

### 1. Create a virtual environment

**Windows (PowerShell):**

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux/macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install the PyTorch and torchaudio builds appropriate for your operating system and CUDA version
if the default pip packages are not suitable for your hardware. See the official PyTorch installation
selector for the correct command.

## Run the project

### FastAPI server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Open the interactive API documentation at `http://127.0.0.1:8000/docs`.

Endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Basic service health check |
| POST | `/translate-audio` | Upload audio and receive translated WAV audio |
| WebSocket | `/ws/audio` | Send audio frames and receive synthesized audio frames |

The HTTP endpoint accepts multipart form fields `audio_file`, `target_lang` (default `hin_Deva`),
and `tts_code` (default `hin`). Transcript and translation text are returned in the
`X-Transcript` and `X-Translation` response headers using URL encoding.

The WebSocket endpoint currently uses Hindi as its target language, matching the original
implementation.

### Local audio-file test

Place a WAV file named `harvard.wav` in the project root, then run:

```bash
python test_local.py
```

Edit `SOURCE_LANGUAGE`, `TARGET_NLLB_LANGUAGE` and `TARGET_TTS_LANGUAGE` at the top of the file
to select the language pair. The generated output is written to `translated_output.wav`.

### Microphone test

```bash
python test_mic.py
```

The script records a short sample, processes it and writes `live_translated_output.wav`.

### Continuous real-time microphone translation

```bash
python realtime_mic.py
```

Speak a sentence and pause to submit it for translation. Say **“stop translation”** or type **q**
and press Enter to stop. Configure the source and target language constants near the top of
`realtime_mic.py` before running.

## Language configuration

The source-language map used by the translation model is defined in `pipeline.py`:

| Short code | NLLB language code |
|---|---|
| `en` | `eng_Latn` |
| `ml` | `mal_Mlym` |
| `hi` | `hin_Deva` |
| `fr` | `fra_Latn` |
| `es` | `spa_Latn` |

The NLLB target language identifier and MMS TTS language suffix are separate values. For example,
Malayalam uses `mal_Mlym` for NLLB and `mal` for MMS TTS; Hindi uses `hin_Deva` and `hin`.

## Performance notes

- The first startup can take time because model files are downloaded and loaded into memory.
- GPU execution is selected automatically when CUDA is available; otherwise CPU mode is used.
- CPU NLLB dynamic quantization is enabled by default in `pipeline.py`.
- TTS models are loaded on demand and cached by language for subsequent requests.

## Development quality checks

Install the optional tools and run the formatter/linter:

```bash
pip install black ruff
black --check .
ruff check .
```

Compile all Python files without loading ML models:

```bash
python -m compileall -q .
```

## Important notes

- The project retains the original ASR -> NLLB translation -> MMS TTS architecture and default model choices.
- This is a development baseline, not a production-hardened public service. Before exposing the API
  to the internet, add authentication, upload-size limits, request timeouts, rate limiting, and
  deployment-specific model-worker management.
- The current source-language detection value is the configured `source_lang`; Faster-Whisper is
  instructed to use that language rather than performing automatic language detection.

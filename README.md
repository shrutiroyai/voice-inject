# voice-inject

A fully local AI voice system for macOS (Apple Silicon). Transcribes speech using the Neural Engine, identifies speakers over time, and pastes clean text into any app — no cloud, no subscriptions.

## Two modes

### Meeting Mode

Open the browser UI at `http://localhost:3000`, click Record, and start talking. The system continuously listens, detects speech with VAD, and streams live transcripts to the browser with speaker labels ("Speaker 1", "Speaker 2"). After the session ends, annotate speakers with real names — this enrolls their voice embeddings so the system recognizes them in future sessions. Transcripts are saved as `.txt` and `.json`.

### Command Mode

Double-tap the left Option key to start recording. Speak naturally. VAD detects pauses and auto-transcribes each segment. A local LLM (Phi-3.5-mini) cleans up the text — fixing grammar, removing filler words, handling self-corrections — then pastes it wherever your cursor is. Double-tap again to stop. Works in any app.

## How it works

```
Microphone (16kHz mono)
    └── WebRTC VAD — detects speech segments
            ├── Meeting Mode
            │     └── pyannote/segmentation — diarizes speakers
            │           └── pyannote/embedding — 512-dim voiceprints
            │                 └── speaker_db.py — centroid matching + adaptive learning
            │                       └── MLX Whisper — transcription → browser UI via WebSocket
            └── Command Mode
                  └── MLX Whisper — transcription
                        └── Phi-3.5-mini-instruct-4bit — LLM text cleanup
                              └── pbpaste / AppleScript — paste into active app
```

Speaker recognition improves over time. The system stores up to 15 embeddings per speaker and updates rolling centroids across sessions.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- HuggingFace account (free) — required for pyannote model access
- ~4GB disk for models

## Install

```bash
git clone https://github.com/shrutiroyai/voice-inject.git
cd voice-inject
bash install.sh
```

## Configuration

Create a `.env` file in the project root:

```
HUGGINGFACE_TOKEN=hf_your_token_here
MIN_SPEECH_ENERGY=50
```

`MIN_SPEECH_ENERGY` controls mic sensitivity. Lower values (e.g. 30) work better for quiet or close mics; raise it (e.g. 100+) in noisy environments.

To get a HuggingFace token: [huggingface.co](https://huggingface.co) → Settings → Access Tokens → New token (read scope). You'll also need to accept the terms for [pyannote/segmentation](https://huggingface.co/pyannote/segmentation) and [pyannote/embedding](https://huggingface.co/pyannote/embedding).

## Speaker enrollment

Speakers can be enrolled two ways:

**Via the browser UI** — after a meeting session, click the annotate panel, assign real names to speaker labels. This enrolls embeddings automatically.

**Via CLI:**

```bash
python enroll_speaker.py enroll "Name"
python enroll_speaker.py list
python enroll_speaker.py test
```

Run `enroll` 2–3 times per speaker for best accuracy.

## Technical stack

| Component | Technology |
|-----------|-----------|
| Transcription | MLX Whisper (`mlx-community/whisper-small-mlx`) |
| Speaker diarization | pyannote/segmentation |
| Speaker identification | pyannote/embedding (512-dim) |
| LLM text cleanup | Phi-3.5-mini-instruct-4bit via mlx-lm |
| VAD | WebRTC VAD |
| Server | FastAPI + WebSocket |
| UI | Vanilla HTML/JS served by FastAPI at localhost:3000 |
| Audio | sounddevice (PortAudio) at 16kHz mono |

## Key files

| File | Purpose |
|------|---------|
| `client.py` | Audio capture, VAD, transcription, command mode, speaker resolution |
| `server.py` | FastAPI server, WebSocket relay, browser UI, annotation API |
| `speaker_id.py` | Embedding extraction, diarization, speaker identification |
| `speaker_db.py` | JSON-based speaker database with centroid matching |
| `enroll_speaker.py` | CLI for speaker enrollment |
| `install.sh` | One-command installer |

## Logs

```
/tmp/voice-inject-server.log
/tmp/voice-inject-client.log
```

## Notes

- English-only transcription
- Whisper hallucination filter blocks common artifacts ("thank you", "thanks for watching", etc.)
- Models download once on first run, then work fully offline
- Nothing leaves your machine at any point

# Project Instructions: Voice Inject

## Core Architecture
- This application is optimized for **Command Mode** only. Meeting transcription, speaker identification, and diarization have been removed for maximum performance and simplicity.
- **Thread Safety:** All MLX operations (Whisper and LLM) must run sequentially on a dedicated `mlx_worker` thread to prevent GPU stream/thread conflicts.
- The speaker database (PostgreSQL/pgvector) has been removed.

## LLM Post-Processing
- The LLM cleanup is strictly for **grammar and punctuation**. It must preserve original wording, tone, and filler words.
- Every transcribed segment must end with a single space to ensure natural flow when batch-pasted.

## Configuration
- Settings are primarily managed in `~/.voice-inject/config.yaml` and are accessible via the UI.
- Support for **Environment Presets** (Laptop, Office, Studio) allows easy adjustment of microphone sensitivity.
- **Hugging Face Token:** Can be configured via the UI to support gated model downloads without silent failures.

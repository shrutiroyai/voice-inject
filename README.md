# Voice Inject 🎙️

AI-powered voice dictation with intelligent text cleaning. Press a hotkey to record, speak naturally, and get cleaned text with filler words removed and grammar fixed automatically.

## Quick Start

### Prerequisites

```bash
# Install ffmpeg
brew install ffmpeg

# Configure AWS credentials (for Bedrock access)
aws configure
```

### Run

```bash
git clone <your-repo>
cd voice-inject
./start.sh
```

Done! Your browser will open automatically.

### Use It

1. Press **Ctrl** (Control key) to start recording
2. Speak your text
3. Press **Ctrl** again to stop
4. Cleaned text appears in your application

Press `Ctrl+C` in the terminal to stop all services.

---

## Features

- **🎤 Voice Capture** - Ctrl key to start/stop recording
- **🤖 AI Cleaning** - AWS Bedrock removes "um", "like", fixes grammar
- **⚙️ Customizable** - Web UI to adjust cleaning level and tone
- **📝 Smart Vocabulary** - Teach it your custom terms and acronyms
- **💬 Command Mode** - Say "Molly" to give LLM commands

## How It Works

Three components running locally:

1. **Server** - FastAPI backend using AWS Bedrock
2. **Client** - Desktop app listening for Ctrl key
3. **UI** - React settings page (auto-opens at http://localhost:5173)

## Configuration

### Customize Text Cleaning

Open http://localhost:5173 to configure:

- **Creativity Level**
  - Level 1: Fix grammar only  
  - Level 2: Make concise
  - Level 3: Rewrite for clarity
- **Tone**: neutral, professional, casual, friendly
- **Vocabulary**: Add custom word replacements

### Train on Your Domain

Edit `config/config.py` to add your frequently used terms:

```python
USER_CONTEXT = """
Software engineer working with AWS, Kubernetes, Docker.
Common terms: API, microservices, CI/CD, pipeline, deployment.
"""
```

This helps Bedrock understand your domain-specific vocabulary.

## Command Mode

Start your dictation with "Molly" to give the LLM instructions:

```
"Molly, make this formal: the numbers are good"
→ "The metrics are performing well."

"Molly, summarize: we had a long meeting about the project..."
→ Concise summary

"Molly, bullet points: first X, then Y, then Z"
→ Formatted list
```

## Troubleshooting

**Voice capture not working?**
```bash
# Check if client is running
ps aux | grep client.py

# Check logs
tail -f /tmp/voice-inject-client.log
```

**Server issues?**
```bash
# Check logs
tail -f /tmp/voice-inject-server.log

# Verify AWS credentials
aws sts get-caller-identity
```

**Port conflicts?**
```bash
lsof -ti :3000 | xargs kill -9
```

## Technical Details

- **Python 3.12+** required
- **Node.js 18+** for UI
- **AWS Bedrock** for transcription and LLM
- Logs: `/tmp/voice-inject-*.log`
- Settings: `~/.voice-inject/config.yaml`

## Development

Run components separately:

```bash
python server.py        # Server only
cd ui && npm run dev    # UI only  
python client.py        # Client only
```

API docs: http://localhost:3000/docs

---

Built with [AWS Bedrock](https://aws.amazon.com/bedrock/), [FastAPI](https://fastapi.tiangolo.com/), and [React](https://react.dev/)

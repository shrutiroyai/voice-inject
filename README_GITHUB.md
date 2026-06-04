# Voice Inject 🎙️

*Hold Control, speak, release — cleaned text appears wherever your cursor is.*

A voice-to-text dictation tool that uses AWS Transcribe for real-time speech recognition and Amazon Bedrock (Claude) for intelligent text cleanup. Built in ~200 lines of Python.

## Features

- **Push-to-talk**: Hold Control key to dictate, release to paste
- **Real-time transcription**: AWS Transcribe streaming for low latency
- **AI cleanup**: Claude Haiku removes filler words, adds punctuation
- **Custom vocabulary**: Define domain-specific terms (jargon, acronyms)
- **Command mode**: Say "Molly" to trigger advanced LLM commands
- **Auto-paste**: Cleaned text automatically pastes into any app

## Architecture

```
Microphone → AWS Transcribe Streaming → Amazon Bedrock → Clipboard → Auto-paste
```

## Prerequisites

- **macOS** (for `pbcopy` and `osascript` paste automation)
- **Python 3.12+**
- **AWS Account** with:
  - AWS Transcribe access
  - Amazon Bedrock access (Claude 3.5 Haiku)
  - Valid AWS credentials configured (`~/.aws/credentials` or SSO)

## Installation

```bash
git clone https://github.com/shrutiroyai/voice-inject.git
cd voice-inject
./install.sh
```

This creates a virtual environment and installs dependencies:
- `amazon-transcribe`
- `boto3`
- `sounddevice`
- `pynput`
- `pyyaml`

## Usage

```bash
python voice_inject_github.py
```

1. **Hold Control** → Start recording
2. **Speak** → Audio streams to AWS Transcribe
3. **Release Control** → Processing begins
4. **Wait ~2 seconds** → Cleaned text auto-pastes

Press **Esc** to quit.

## Custom Vocabulary

Edit `~/.voice-inject/vocab.yaml` to add domain-specific terms:

```yaml
corrections:
  - hear: ["year over year", "y o y", "yoy"]
    use: "YoY"
  - hear: ["gen ai", "jen ai"]
    use: "GenAI"
  - hear: ["t y", "tv", "tee why"]
    use: "TY"
```

No restart needed — changes apply immediately.

## Two Modes

### Normal Mode (Default)
Light editing only:
- Remove filler words (um, uh, like)
- Add punctuation
- Capitalize properly
- Apply vocabulary corrections

**Example:**
- Input: *"um so like the y o y numbers look good"*
- Output: *"The YoY numbers look good."*

### Command Mode (Say "Molly")
Full LLM assistance for transformations:

- *"Molly, make this more formal: the numbers look good"* → *"The metrics are performing well."*
- *"Molly, summarize this: we had a meeting about..."* → concise summary
- *"Molly, rewrite as bullet points: first X then Y"* → formatted list

## Cost Analysis

**For 100 dictations/day (avg 20 seconds each):**
- AWS Transcribe Streaming: ~$33/month
- Bedrock Claude 3.5 Haiku: ~$3/month
- **Total: ~$36/month**

## Troubleshooting

### Auto-paste fails
If you see: `⚠️ Auto-paste failed (use Cmd+V manually)`

**Cause:** macOS accessibility permissions or keyboard state collision

**Solution:** Text is always in clipboard — just press Cmd+V manually

### Credentials expired
Ensure AWS credentials are valid:
```bash
aws sts get-caller-identity
```

For AWS SSO users, refresh before running:
```bash
aws sso login
```

## Project Structure

```
voice-inject/
├── voice_inject_github.py  # Main application
├── vocab.py                 # Vocabulary management
├── requirements.txt         # Python dependencies
├── install.sh              # Setup script
└── default_vocab.yaml      # Sample vocabulary file
```

## How It Works

1. **Audio Capture**: `sounddevice` captures 16kHz PCM audio in 100ms chunks
2. **Streaming Transcription**: Audio chunks stream to AWS Transcribe via WebSocket
3. **AI Cleanup**: Raw transcript sent to Bedrock Claude for cleanup
4. **Auto-paste**: Cleaned text copied to clipboard + auto-pasted via `osascript`

## Contributing

This is a personal project, but contributions welcome! Feel free to:
- Open issues for bugs or feature requests
- Submit PRs with improvements
- Fork and customize for your needs

## License

MIT License - See LICENSE file for details

## Disclaimer

This is a personal project built using publicly available AWS services. All opinions and views are my own.

## Acknowledgments

Built with:
- AWS Transcribe Streaming
- Amazon Bedrock (Claude 3.5 Haiku)
- Python `sounddevice`, `pynput`, `boto3`

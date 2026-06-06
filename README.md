# 🎙️ Voice Inject

**Fast, accurate voice-to-text dictation for macOS using OpenAI Whisper.**

Press Control, speak your text, release Control — and watch it appear instantly in any application.

## ✨ Features

- **Local faster-whisper transcription** - No cloud dependencies, works offline
- **Auto-paste** - Text appears directly in your active application
- **Web UI** - Settings dashboard at localhost:5173
- **One-command install** - Run `./install.sh` and you're done
- **Quick launch** - Type `voice` from anywhere after install

## 📋 Requirements

- macOS (tested on macOS 13+)
- Python 3.9+
- Node.js
- ffmpeg (`brew install ffmpeg`)
- Microphone access permissions

## 🚀 Quick Start

One command installs everything and opens the browser:

```bash
./install.sh
```

That's it. The installer handles Python dependencies, Node dependencies, configuration, and launches all services automatically. Your browser will open to the Voice Inject UI when ready.

### Returning Users

After your first install, just type `voice` in any terminal to launch:

```bash
voice
```

The `voice` alias is registered automatically during install. If you just installed, restart your terminal or run `source ~/.zshrc` for the alias to take effect.

## 🎯 Usage

1. Click into any text field
2. **Hold Control** (left or right)
3. Speak your text
4. **Release Control**
5. Text appears automatically! 🎉

Press `Ctrl+C` in the installer terminal to stop all services cleanly.

## 📁 Project Structure

```
voice-inject/
├── install.sh          # One-command installer & launcher
├── client.py           # faster-whisper transcription + auto-paste
├── server.py           # WebSocket server + settings API
├── requirements.txt    # Python dependencies
└── ui/                 # Web UI (React + Vite)
```

## 🎯 Optimizing Performance

### Whisper Model Selection

Edit `client.py` to change the model (line with `WhisperModel`):

| Model | Size | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| tiny | 39MB | 🚀 ~0.2s | ⭐⭐ | Quick notes |
| base | 74MB | 🚀 ~0.3s | ⭐⭐⭐ | Casual use |
| **small** | 244MB | 🏃 ~0.5s | ⭐⭐⭐⭐ | **Default — good balance** |
| medium | 1.5GB | 🐢 ~1.5s | ⭐⭐⭐⭐⭐ | Max accuracy |

```python
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
```

The app uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2) which is 4x faster than OpenAI's original Whisper implementation.

## 🔧 Troubleshooting

### "Cannot connect to server"
- Make sure services are running via `./install.sh` or `voice`
- Check port 3000 is not in use: `lsof -ti :3000`

### "No audio recorded" or Audio Errors
- Check System Preferences → Sound → Input
- Grant microphone permissions to Terminal
- Try restarting the client

### Whisper Model Download
First run downloads the `small` model (~244MB). This only happens once.

### Service Logs
If something goes wrong, check the log files:
- Server: `/tmp/voice-inject-server.log`
- UI: `/tmp/voice-inject-ui.log`
- Client: `/tmp/voice-inject-client.log`


<details>
<summary><h2>🛠️ Advanced: Manual Setup</h2></summary>

If you prefer to set things up manually instead of using the one-command installer:

### 1. Install Dependencies

```bash
# Install system tools
brew install ffmpeg

# Create Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Install Node dependencies
cd ui && npm install && cd ..
```

### 2. Run Services

**Terminal 1 - Start Server:**
```bash
source .venv/bin/activate
python server.py
```

**Terminal 2 - Start UI:**
```bash
cd ui && npm run dev
```

**Terminal 3 - Start Client:**
```bash
source .venv/bin/activate
python client.py
```

### 4. Use It

1. Open http://localhost:5173 in your browser
2. Click into any text field
3. **Hold Control** (left or right)
4. Speak your text
5. **Release Control**
6. Text appears automatically! 🎉

Press `Esc` in the client terminal to quit.

### Background Mode with tmux

Add to your `~/.zshrc`:

```bash
voice() {
  if tmux has-session -t voice 2>/dev/null; then
    echo "🎤 Voice client already running"
    return 0
  fi
  
  echo "🚀 Starting voice-inject client in background..."
  tmux new-session -d -s voice "cd /path/to/voice-inject && python client.py"
  sleep 2
  echo "✅ Client running in background!"
}

voice-stop() {
  tmux kill-session -t voice 2>/dev/null
  echo "✅ Voice client stopped"
}
```

Then just run `voice` to start!

</details>

## 📝 How It Works

1. **Client** captures audio when Control is held
2. **faster-whisper** (local) transcribes audio to text
3. **Client** auto-pastes the transcribed text into your active app

All processing happens locally — no cloud API calls required!

## 🤝 Contributing

Contributions welcome! Please:
1. Test your changes thoroughly
2. Update documentation
3. Follow existing code style

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- Built for developers who type too much

---

**Pro tip:** Use the `small` model in `client.py` for faster transcription if accuracy isn't critical!

## Speaker Identification

Identify who is speaking during transcription using on-device speaker embeddings (pyannote.audio). All processing is local — no audio leaves the machine.

### 1. Install dependencies

```bash
pip install pyannote.audio scipy torch
```

### 2. Get a HuggingFace token

1. Go to [huggingface.co](https://huggingface.co) → Settings → Access Tokens → New token (read scope)
2. Accept the model terms at [huggingface.co/pyannote/embedding](https://huggingface.co/pyannote/embedding)

### 3. Set the token

Either export it in your shell:

```bash
export HUGGINGFACE_TOKEN=hf_xxx
```

Or add it to `~/.voice-inject/config.yaml`:

```yaml
hf_token: hf_xxx
```

### 4. Enroll a speaker

```bash
python enroll_speaker.py enroll "Your Name"
```

Run 2–3 times for best accuracy.

### 5. List enrolled speakers

```bash
python enroll_speaker.py list
```

### 6. Test identification

```bash
python enroll_speaker.py test
```

> The model downloads once (~250MB) on first use, then runs fully offline.

# 🎙️ Voice Inject

**Fast, accurate voice-to-text dictation for macOS using OpenAI Whisper.**

Press Control, speak your text, release Control — and watch it appear instantly in any application.

## ✨ Features

- **Local Whisper transcription** - No cloud dependencies, works offline
- **Smart vocabulary corrections** - Teach it your custom terminology
- **Auto-paste** - Text appears directly in your active application
- **Simple server** - Basic text cleaning and vocabulary replacement
- **Lightweight** - Runs in background via tmux

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
├── client.py           # Whisper transcription + auto-paste
├── server.py           # Text cleaning server
├── requirements.txt    # Python dependencies
├── ui/                 # Web UI (React + Vite)
└── ~/.voice-inject/    # Runtime config directory (vocab, settings)
```

## ⚙️ Configuration

### Vocabulary Corrections

Add custom vocabulary in `~/.voice-inject/vocab.yaml`:

```yaml
corrections:
  - hear: ["kubernetes", "cuber netties"]
    use: "Kubernetes"
  - hear: ["react", "re act"]
    use: "React"
```

## 🎯 Optimizing Performance

### Whisper Model Selection

Edit `client.py` line 46 to change model:

| Model | Size | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| tiny | 39MB | 🚀 0.3s | ⭐⭐ | Quick notes |
| base | 74MB | 🚀 0.5s | ⭐⭐⭐ | Casual use |
| small | 244MB | 🏃 0.8s | ⭐⭐⭐⭐ | Good balance |
| **medium** | 1.5GB | 🐢 1.5s | ⭐⭐⭐⭐⭐ | **Default - best accuracy** |
| large | 2.9GB | 🐌 3s+ | ⭐⭐⭐⭐⭐ | Overkill |

```python
whisper_model = whisper.load_model("small")  # Faster alternative
```

## 🔧 Troubleshooting

### "Cannot connect to server"
- Make sure services are running via `./install.sh` or `voice`
- Check port 3000 is not in use: `lsof -ti :3000`

### "No audio recorded" or Audio Errors
- Check System Preferences → Sound → Input
- Grant microphone permissions to Terminal
- Try restarting the client

### Whisper Model Download
First run downloads ~1.5GB model. Be patient!

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
2. **Whisper** (local) transcribes audio to text
3. **Server** applies vocabulary corrections
4. **Client** auto-pastes cleaned text

All processing happens locally - no cloud API calls required!

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

**Pro tip:** Add your domain-specific vocabulary to `~/.voice-inject/vocab.yaml` for best results!

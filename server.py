#!/usr/bin/env python3
"""Voice Inject Server — simple text processing with WebSocket support."""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import yaml
import sys
import logging
from pathlib import Path
from datetime import datetime
import json

# Import user context
try:
    from config.config import USER_CONTEXT
except ImportError:
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "config_example",
        Path(__file__).parent / "config" / "config.example.py"
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    USER_CONTEXT = _mod.USER_CONTEXT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/tmp/voice-inject-server.log', mode='w')
    ],
    force=True
)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config paths
CONFIG_DIR = Path.home() / ".voice-inject"
CONFIG_PATH = CONFIG_DIR / "config.yaml"
VOCAB_PATH = CONFIG_DIR / "vocab.yaml"
TRANSCRIPTS_DIR = CONFIG_DIR / "transcripts"

CONFIG_DIR.mkdir(exist_ok=True)
TRANSCRIPTS_DIR.mkdir(exist_ok=True)

# WebSocket connections
active_connections = []


def load_config():
    """Load configuration from config.yaml."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    return {
        "save_transcripts": False,
        "creativity_level": 0.3,
        "tone": "neutral"
    }


def save_config(config: dict):
    """Save configuration to config.yaml."""
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def load_vocab():
    """Load vocabulary from vocab.yaml."""
    if VOCAB_PATH.exists():
        with open(VOCAB_PATH) as f:
            data = yaml.safe_load(f) or {}
            corrections = data.get("corrections", [])
            logger.info(f"Loaded {len(corrections)} vocab corrections from {VOCAB_PATH}")
            return corrections
    logger.info(f"Vocab file not found at {VOCAB_PATH}")
    return []


def save_vocab(corrections: list):
    """Save vocabulary to vocab.yaml."""
    with open(VOCAB_PATH, "w") as f:
        yaml.dump({"corrections": corrections}, f, default_flow_style=False, allow_unicode=True)


def apply_vocab_corrections(text: str) -> str:
    """Apply vocabulary corrections to text."""
    corrections = load_vocab()
    result = text
    
    for entry in corrections:
        target = entry.get("use", "")
        variants = entry.get("hear", [])
        
        for variant in variants:
            # Case-insensitive replacement
            import re
            pattern = re.compile(re.escape(variant), re.IGNORECASE)
            result = pattern.sub(target, result)
    
    return result


def save_transcript(text: str) -> str:
    """Save transcript to file and return filepath."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"transcript_{timestamp}.txt"
    filepath = TRANSCRIPTS_DIR / filename
    
    with open(filepath, "w") as f:
        f.write(f"# Voice Inject Transcript\n")
        f.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(text)
    
    logger.info(f"Saved transcript to {filepath}")
    return str(filepath)


def basic_clean(text: str) -> str:
    """Basic text cleaning without LLM - just apply vocab corrections."""
    if not text.strip():
        return ""
    
    # Apply vocabulary corrections
    cleaned = apply_vocab_corrections(text)
    
    # Basic cleanup: capitalize first letter, ensure punctuation
    cleaned = cleaned.strip()
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    
    if cleaned and cleaned[-1] not in '.!?':
        cleaned += '.'
    
    logger.info(f"Basic clean: '{text}' -> '{cleaned}'")
    return cleaned


# API Models
class CleanRequest(BaseModel):
    text: str
    creativity_level: int = None
    tone: str = None


class CleanResponse(BaseModel):
    cleaned_text: str


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication with client and UI."""
    await websocket.accept()
    active_connections.append(websocket)
    logger.info("WebSocket client connected")
    
    try:
        while True:
            # Receive messages from client/UI
            data = await websocket.receive_text()
            message = json.loads(data)
            logger.info(f"📥 Received WebSocket message: {message}")
            
            # Handle different message types
            if message.get("type") == "toggle_recording":
                # Broadcast to all connected clients
                for connection in active_connections:
                    if connection != websocket:
                        await connection.send_text(json.dumps(message))
            
            elif message.get("type") == "status":
                # Broadcast recording status to all other clients (browser UIs)
                for connection in active_connections:
                    if connection != websocket:
                        await connection.send_text(data)
            
            elif message.get("type") == "transcript":
                # Handle transcript - save if enabled
                config = load_config()
                text = message.get("text", "")
                
                if config.get("save_transcripts", False):
                    filepath = save_transcript(text)
                    await websocket.send_text(json.dumps({
                        "type": "transcript_saved",
                        "filepath": filepath
                    }))
                
                # Broadcast to all UIs
                for connection in active_connections:
                    if connection != websocket:
                        await connection.send_text(data)
            
            elif message.get("type") == "config_update":
                # Update configuration
                config = load_config()
                config.update(message.get("config", {}))
                save_config(config)
                
                # Broadcast to all clients
                for connection in active_connections:
                    await connection.send_text(json.dumps({
                        "type": "config_updated",
                        "config": config
                    }))
    
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)


@app.post("/api/clean", response_model=CleanResponse)
async def clean_text(request: CleanRequest):
    """Clean text with basic processing (no LLM)."""
    logger.info(f"Received clean request: text='{request.text[:50]}...'")
    
    if not request.text.strip():
        return CleanResponse(cleaned_text="")
    
    # Just apply vocab corrections and basic cleanup
    cleaned = basic_clean(request.text)
    
    logger.info(f"Got result: '{cleaned[:50]}...'")
    
    return CleanResponse(cleaned_text=cleaned)


@app.get("/api/config")
async def get_config():
    """Get current configuration."""
    return load_config()


@app.post("/api/config")
async def update_config(config: dict):
    """Update configuration."""
    save_config(config)
    
    # Broadcast to WebSocket clients
    for connection in active_connections:
        await connection.send_text(json.dumps({
            "type": "config_updated",
            "config": config
        }))
    
    return {"success": True}


@app.get("/api/vocab")
async def get_vocab():
    """Get vocabulary rules."""
    corrections = load_vocab()
    return {"corrections": corrections}


@app.post("/api/vocab")
async def update_vocab(data: dict):
    """Update vocabulary rules."""
    corrections = data.get("corrections", [])
    save_vocab(corrections)
    return {"success": True}


@app.get("/api/transcripts")
async def list_transcripts():
    """List all saved transcripts."""
    transcripts = []
    for filepath in TRANSCRIPTS_DIR.glob("transcript_*.txt"):
        stat = filepath.stat()
        transcripts.append({
            "name": filepath.name,
            "path": str(filepath),
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
        })
    
    # Sort by modified time, most recent first
    transcripts.sort(key=lambda x: x["modified"], reverse=True)
    return {"transcripts": transcripts}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "voice-inject-server"}


# Simple HTML UI
@app.get("/", response_class=HTMLResponse)
async def get_ui():
    """Serve simple HTML UI."""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Voice Inject</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 20px;
            padding: 40px;
            width: 100%;
            max-width: 600px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 10px;
            font-size: 32px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 14px;
        }
        .record-btn {
            width: 150px;
            height: 150px;
            border-radius: 50%;
            border: none;
            font-size: 60px;
            cursor: pointer;
            display: block;
            margin: 0 auto 30px;
            transition: all 0.3s;
            background: #f0f0f0;
            color: #666;
        }
        .record-btn.recording {
            background: #ff4444;
            color: white;
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
        .status {
            text-align: center;
            font-size: 18px;
            color: #666;
            margin-bottom: 20px;
            min-height: 30px;
        }
        .toggle-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px;
            background: #f9f9f9;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .toggle-label {
            font-weight: 600;
            color: #333;
        }
        .toggle {
            position: relative;
            width: 60px;
            height: 30px;
        }
        .toggle input {
            opacity: 0;
            width: 0;
            height: 0;
        }
        .slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #ccc;
            transition: 0.4s;
            border-radius: 30px;
        }
        .slider:before {
            position: absolute;
            content: "";
            height: 22px;
            width: 22px;
            left: 4px;
            bottom: 4px;
            background-color: white;
            transition: 0.4s;
            border-radius: 50%;
        }
        input:checked + .slider {
            background-color: #667eea;
        }
        input:checked + .slider:before {
            transform: translateX(30px);
        }
        .transcript-box {
            background: #f9f9f9;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            padding: 20px;
            min-height: 150px;
            max-height: 300px;
            overflow-y: auto;
            font-size: 16px;
            line-height: 1.6;
            color: #333;
            margin-bottom: 20px;
            white-space: pre-wrap;
        }
        .transcript-box:empty:before {
            content: "Transcript will appear here...";
            color: #999;
        }
        .buttons {
            display: flex;
            gap: 10px;
            justify-content: center;
        }
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.2s;
            font-weight: 600;
        }
        .btn-copy {
            background: #667eea;
            color: white;
        }
        .btn-copy:hover {
            background: #5568d3;
        }
        .btn-clear {
            background: #f0f0f0;
            color: #666;
        }
        .btn-clear:hover {
            background: #e0e0e0;
        }
        .hotkey-hint {
            text-align: center;
            color: #999;
            font-size: 12px;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ Voice Inject</h1>
        <p class="subtitle">faster-whisper edition</p>
        
        <button class="record-btn" id="recordBtn">⏺️</button>
        <div class="status" id="status">Ready</div>
        
        <div class="toggle-container">
            <span class="toggle-label">Save Transcripts</span>
            <label class="toggle">
                <input type="checkbox" id="saveToggle">
                <span class="slider"></span>
            </label>
        </div>
        
        <div class="transcript-box" id="transcript"></div>
        
        <div class="buttons">
            <button class="btn btn-copy" id="copyBtn">Copy All</button>
            <button class="btn btn-clear" id="clearBtn">Clear</button>
        </div>
        
        <p class="hotkey-hint">Double-tap Right Option (⌥) to toggle recording from anywhere</p>
    </div>
    
    <script>
        let ws = null;
        let isRecording = false;
        let transcript = "";
        
        const recordBtn = document.getElementById('recordBtn');
        const status = document.getElementById('status');
        const transcriptBox = document.getElementById('transcript');
        const saveToggle = document.getElementById('saveToggle');
        const copyBtn = document.getElementById('copyBtn');
        const clearBtn = document.getElementById('clearBtn');
        
        // Connect to WebSocket
        function connectWebSocket() {
            ws = new WebSocket('ws://localhost:3000/ws');
            
            ws.onopen = () => {
                console.log('Connected to server');
                // Request current config
                fetch('/api/config')
                    .then(r => r.json())
                    .then(config => {
                        saveToggle.checked = config.save_transcripts || false;
                    });
            };
            
            ws.onmessage = (event) => {
                const message = JSON.parse(event.data);
                
                if (message.type === 'status') {
                    isRecording = message.recording;
                    updateUI();
                } else if (message.type === 'transcript') {
                    if (transcript) transcript += '\\n\\n';
                    transcript += message.text;
                    transcriptBox.textContent = transcript;
                    transcriptBox.scrollTop = transcriptBox.scrollHeight;
                } else if (message.type === 'transcript_saved') {
                    status.textContent = `✅ Saved to ${message.filepath}`;
                }
            };
            
            ws.onclose = () => {
                console.log('Disconnected, reconnecting in 2s...');
                setTimeout(connectWebSocket, 2000);
            };
        }
        
        function updateUI() {
            if (isRecording) {
                recordBtn.textContent = '⏹️';
                recordBtn.classList.add('recording');
                status.textContent = 'Recording...';
            } else {
                recordBtn.textContent = '⏺️';
                recordBtn.classList.remove('recording');
                status.textContent = 'Ready';
            }
        }
        
        recordBtn.addEventListener('click', () => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'toggle_recording' }));
            }
        });
        
        saveToggle.addEventListener('change', () => {
            const config = { save_transcripts: saveToggle.checked };
            fetch('/api/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });
        });
        
        copyBtn.addEventListener('click', () => {
            navigator.clipboard.writeText(transcript)
                .then(() => {
                    status.textContent = '✅ Copied to clipboard';
                    setTimeout(() => status.textContent = 'Ready', 2000);
                });
        });
        
        clearBtn.addEventListener('click', () => {
            transcript = "";
            transcriptBox.textContent = "";
        });
        
        connectWebSocket();
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    print("🎙️  Voice Inject Server starting...")
    print("   Mode: Simple (no LLM)")
    print("   API: http://localhost:3000")
    print("   UI: http://localhost:3000")
    print("   WebSocket: ws://localhost:3000/ws")
    print("   Endpoints:")
    print("     POST /api/clean - Basic text cleaning")
    print("     GET/POST /api/config - Settings")
    print("     GET/POST /api/vocab - Vocabulary")
    print("     GET /health - Health check")
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=3000, log_level="info")

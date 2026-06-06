#!/usr/bin/env python3
"""Voice Inject Server — simple text processing with WebSocket support."""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import yaml
import sys
import logging
from pathlib import Path
from datetime import datetime
import json
import wave
import numpy as np
from fastapi import HTTPException
from fastapi.responses import FileResponse


def _load_dotenv():
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

import os
_load_dotenv()


_speaker_db = None
_speaker_identifier = None


def get_speaker_db():
    global _speaker_db
    if _speaker_db is None:
        from speaker_db import SpeakerDB
        _speaker_db = SpeakerDB()
    return _speaker_db


def get_speaker_identifier():
    global _speaker_identifier
    if _speaker_identifier is None:
        from speaker_id import SpeakerIdentifier
        _speaker_identifier = SpeakerIdentifier()
    return _speaker_identifier

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
# Config paths
CONFIG_DIR = Path.home() / ".voice-inject"
CONFIG_PATH = CONFIG_DIR / "config.yaml"
SCRIPT_DIR = Path(__file__).parent.resolve()
TRANSCRIPTS_DIR = SCRIPT_DIR / "transcripts"

CONFIG_DIR.mkdir(exist_ok=True)
TRANSCRIPTS_DIR.mkdir(exist_ok=True)

# WebSocket connections
active_connections = []

# Session state: stores the most recent session_started message for late-connecting clients
session_state = {}
# Warmup state: sent to browsers that connect after warmup_started fires
warmup_state = {"type": "warmup_started"}  # default: assume not yet warm


def load_config():
    """Load configuration from config.yaml."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            data = yaml.safe_load(f) or {}
            return data
    return {"save_transcripts": False}


def save_config(config: dict):
    """Save configuration to config.yaml."""
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication with client and UI."""
    global session_state, warmup_state
    await websocket.accept()
    active_connections.append(websocket)
    logger.info("WebSocket client connected")

    # Send warmup state so browsers connecting after warmup_started still see it
    await websocket.send_text(json.dumps(warmup_state))
    # Send stored session_started to late-connecting clients
    if session_state:
        await websocket.send_text(json.dumps(session_state))

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            logger.info(f"📥 Received WebSocket message: {message}")
            
            if message.get("type") == "toggle_recording":
                for connection in active_connections:
                    if connection != websocket:
                        await connection.send_text(json.dumps(message))
            
            elif message.get("type") == "status":
                for connection in active_connections:
                    if connection != websocket:
                        await connection.send_text(data)
            
            elif message.get("type") == "transcript":
                config = load_config()
                text = message.get("text", "")
                
                if config.get("save_transcripts", False):
                    filepath = save_transcript(text)
                    await websocket.send_text(json.dumps({
                        "type": "transcript_saved",
                        "filepath": filepath
                    }))
                
                for connection in active_connections:
                    if connection != websocket:
                        await connection.send_text(data)
            
            elif message.get("type") == "config_update":
                config = load_config()
                config.update(message.get("config", {}))
                save_config(config)
                
                for connection in active_connections:
                    await connection.send_text(json.dumps({
                        "type": "config_updated",
                        "config": config
                    }))

            elif message.get("type") == "session_started":
                session_state = message
                for connection in active_connections:
                    if connection != websocket:
                        await connection.send_text(data)

            elif message.get("type") == "transcript_segment":
                for connection in active_connections:
                    if connection != websocket:
                        await connection.send_text(data)

            elif message.get("type") == "session_ended":
                session_state = {}
                for connection in active_connections:
                    if connection != websocket:
                        await connection.send_text(data)

            elif message.get("type") in ("warmup_started", "warmup_complete"):
                warmup_state = message
                for connection in active_connections:
                    if connection != websocket:
                        await connection.send_text(data)

    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)


@app.get("/api/config")
async def get_config():
    """Get current configuration."""
    return load_config()


@app.post("/api/config")
async def update_config(new_config: dict):
    """Update configuration (merge with existing)."""
    config = load_config()
    config.update(new_config)
    save_config(config)
    
    for connection in active_connections:
        await connection.send_text(json.dumps({
            "type": "config_updated",
            "config": config
        }))
    
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


@app.get("/api/transcripts/path")
async def transcripts_path():
    """Return the transcripts folder path."""
    return {"path": str(TRANSCRIPTS_DIR)}


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
        .hotkey-hint {
            text-align: center;
            color: #999;
            font-size: 12px;
            margin-top: 20px;
        }
        .diagnostics {
            margin-top: 20px;
            padding: 15px;
            border-radius: 10px;
            font-size: 13px;
            line-height: 1.6;
        }
        .diagnostics.ok {
            background: #e8f5e9;
            color: #2e7d32;
        }
        .diagnostics.warning {
            background: #fff3e0;
            color: #e65100;
        }
        .diagnostics.error {
            background: #ffebee;
            color: #c62828;
        }
        .diagnostics h3 {
            margin: 0 0 8px 0;
            font-size: 14px;
        }
        .diagnostics ul {
            margin: 0;
            padding-left: 20px;
        }
        .diagnostics li {
            margin-bottom: 4px;
        }
        .status-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 6px;
        }
        .status-dot.green { background: #4CAF50; }
        .status-dot.yellow { background: #FF9800; }
        .status-dot.red { background: #f44336; }

        /* Live Transcript Styles */
        .live-transcript-section {
            margin-bottom: 30px;
        }
        .live-transcript-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 10px;
        }
        .live-transcript-header h2 {
            font-size: 18px;
            color: #333;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .session-status {
            display: flex;
            align-items: center;
            font-size: 12px;
            color: #666;
            gap: 5px;
        }
        .session-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #ccc;
        }
        .session-dot.active {
            background: #4CAF50;
            animation: pulse-dot 1.5s infinite;
        }
        .session-dot.ended {
            background: #999;
        }
        @keyframes pulse-dot {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .live-transcript {
            background: #f9f9f9;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            padding: 15px;
            min-height: 200px;
            max-height: 300px;
            overflow-y: auto;
            font-size: 14px;
            line-height: 1.6;
            color: #333;
        }
        .live-transcript:empty:before {
            content: "Live transcript will appear here when a session is active...";
            color: #999;
            font-style: italic;
        }
        .segment {
            padding: 6px 0;
            border-bottom: 1px solid #eee;
        }
        .segment:last-child {
            border-bottom: none;
        }
        .segment .time {
            font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
            color: #667eea;
            margin-right: 8px;
            font-size: 13px;
        }
        .section-separator {
            border: none;
            border-top: 2px dashed #e0e0e0;
            margin: 25px 0;
        }
        .disconnection-indicator {
            display: none;
            text-align: center;
            padding: 8px;
            background: #fff3e0;
            border-radius: 6px;
            font-size: 12px;
            color: #e65100;
            margin-top: 8px;
        }
        .disconnection-indicator.visible {
            display: block;
        }
        .tabs {
            display: flex;
            gap: 0;
            margin-bottom: 25px;
            border-radius: 10px;
            overflow: hidden;
            border: 2px solid #667eea;
        }
        .tab {
            flex: 1;
            padding: 12px 20px;
            border: none;
            background: white;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            color: #667eea;
            transition: all 0.2s;
        }
        .tab.active {
            background: #667eea;
            color: white;
        }
        .tab:hover:not(.active) {
            background: #f0f4ff;
        }
        .tab-panel {
            display: none;
        }
        .tab-panel.active {
            display: block;
        }
        .command-info {
            text-align: center;
            padding: 40px 20px;
        }
        .command-icon {
            font-size: 48px;
            margin-bottom: 15px;
            background: #f0f4ff;
            width: 80px;
            height: 80px;
            line-height: 80px;
            border-radius: 50%;
            margin: 0 auto 15px;
            color: #667eea;
        }
        .command-info h2 {
            color: #333;
            margin-bottom: 10px;
        }
        .command-info p {
            color: #666;
            max-width: 350px;
            margin: 0 auto 20px;
            line-height: 1.5;
        }
        .command-status {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 8px 16px;
            background: #f9f9f9;
            border-radius: 20px;
            font-size: 13px;
            color: #666;
        }
        /* Annotate tab */
        .annotate-empty {
            text-align: center;
            padding: 40px 20px;
            color: #999;
            font-size: 14px;
        }
        .annotate-header {
            font-size: 13px;
            color: #666;
            margin-bottom: 12px;
        }
        .annotate-header strong { color: #333; }
        .unknown-list {
            display: flex;
            flex-direction: column;
            gap: 10px;
            max-height: 340px;
            overflow-y: auto;
        }
        .unknown-row {
            background: #f9f9f9;
            border: 1.5px solid #e0e0e0;
            border-radius: 10px;
            padding: 10px 12px;
            transition: border-color 0.3s, background 0.3s;
        }
        .unknown-row.identified {
            border-color: #4CAF50;
            background: #f0fff4;
        }
        .unknown-row-top {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
        }
        .unknown-time {
            font-family: monospace;
            font-size: 11px;
            color: #667eea;
            white-space: nowrap;
        }
        .unknown-text {
            font-size: 13px;
            color: #333;
            flex: 1;
        }
        .speaker-badge {
            font-size: 11px;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 10px;
            white-space: nowrap;
        }
        .unknown-row-bottom {
            display: flex;
            gap: 6px;
        }
        .name-input {
            flex: 1;
            padding: 6px 10px;
            border: 1.5px solid #ddd;
            border-radius: 6px;
            font-size: 13px;
            outline: none;
        }
        .name-input:focus { border-color: #667eea; }
        .assign-btn {
            padding: 6px 14px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            white-space: nowrap;
        }
        .assign-btn:hover { background: #5568d4; }
        .assign-btn:disabled { background: #ccc; cursor: default; }
        .play-btn {
            background: none;
            border: none;
            cursor: pointer;
            font-size: 14px;
            padding: 0 4px;
            color: #667eea;
        }
        .annotate-badge-dot {
            display: none;
            width: 8px;
            height: 8px;
            background: #ff4444;
            border-radius: 50%;
            position: absolute;
            top: 6px;
            right: 6px;
        }
        #tabAnnotate { position: relative; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ Voice Inject</h1>
        
        <!-- Tab Switcher -->
        <div class="tabs">
            <button class="tab active" id="tabMeeting" onclick="switchTab('meeting')">📝 Meeting Mode</button>
            <button class="tab" id="tabCommand" onclick="switchTab('command')">⌨️ Command Mode</button>
            <button class="tab" id="tabAnnotate" onclick="switchTab('annotate')">🏷️ Annotate<span class="annotate-badge-dot" id="annotateDot"></span></button>
        </div>
        
        <!-- Meeting Mode Panel -->
        <div class="tab-panel active" id="panelMeeting">
            <div class="live-transcript-header">
                <div class="session-status" id="sessionStatus">
                    <span class="session-dot" id="sessionDot"></span>
                    <span id="sessionStatusText">Not recording</span>
                </div>
            </div>
            
            <button class="record-btn" id="recordBtn" disabled style="opacity:0.4">⏺️</button>
            <div class="status" id="status">Warming up AI models…</div>
            
            <div class="live-transcript" id="liveTranscript"></div>
            <div class="disconnection-indicator" id="disconnectionIndicator">
                ⚠️ Connection lost — reconnecting...
            </div>
        </div>
        
        <!-- Command Mode Panel -->
        <div class="tab-panel" id="panelCommand">
            <div class="command-info">
                <div class="command-icon">⌥</div>
                <h2>Double-tap Left Option</h2>
                <p>Tap twice to start recording. Tap twice again to stop, transcribe, and auto-paste into your active app.</p>
                <div class="command-status" id="commandStatus">
                    <span class="status-dot green"></span> Ready — waiting for hotkey
                </div>
            </div>
        </div>
        
        <!-- Annotate Panel (meeting mode only) -->
        <div class="tab-panel" id="panelAnnotate">
            <div id="annotateContent">
                <div class="annotate-empty">Complete a meeting session to annotate speakers.</div>
            </div>
        </div>

        <!-- Diagnostics (shared) -->
        <div class="diagnostics" id="diagnostics">
            <h3>⏳ Connecting...</h3>
        </div>
    </div>
    
    <script>
        let ws = null;
        let isRecording = false;
        let clientConnected = false;
        let serverConnected = false;
        let lastClientMessage = 0;
        let userScrolledUp = false;
        let sessionStartTime = null;
        let sessionElapsedInterval = null;
        let currentTab = 'meeting';
        let currentSessionId = null;

        function switchTab(tab) {
            currentTab = tab;
            ['meeting', 'command', 'annotate'].forEach(t => {
                document.getElementById('tab' + t.charAt(0).toUpperCase() + t.slice(1)).classList.toggle('active', t === tab);
                document.getElementById('panel' + t.charAt(0).toUpperCase() + t.slice(1)).classList.toggle('active', t === tab);
            });
            if (tab === 'annotate') {
                document.getElementById('annotateDot').style.display = 'none';
            }
        }

        function speakerColor(name) {
            let h = 0;
            for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) & 0xffff;
            const hue = h % 360;
            return { bg: `hsl(${hue},60%,88%)`, fg: `hsl(${hue},50%,35%)` };
        }

        function loadAnnotateTab(sessionId) {
            currentSessionId = sessionId;
            fetch('/api/sessions/' + sessionId)
                .then(r => r.json())
                .then(session => renderAnnotatePanel(session))
                .catch(() => {
                    document.getElementById('annotateContent').innerHTML =
                        '<div class="annotate-empty">Could not load session data.</div>';
                });
        }

        function needsAnnotation(seg) {
            return !seg.speaker || /^Speaker \\d+$/.test(seg.speaker);
        }

        function renderAnnotatePanel(session) {
            const unknown = (session.segments || []).filter(needsAnnotation);
            const known = (session.segments || []).filter(s => !needsAnnotation(s)).length;
            const total = (session.segments || []).length;
            const content = document.getElementById('annotateContent');

            if (unknown.length === 0) {
                content.innerHTML = '<div class="annotate-empty">All ' + total + ' speakers identified!</div>';
                return;
            }

            let html = '<div class="annotate-header">'
                + '<strong>' + unknown.length + ' unidentified segment' + (unknown.length !== 1 ? 's' : '') + '</strong>'
                + ' &nbsp;·&nbsp; ' + known + ' of ' + total + ' identified'
                + '</div><div class="unknown-list">';

            unknown.forEach(seg => {
                const autoLabel = seg.speaker ? escHtml(seg.speaker) : 'unknown';
                html += `<div class="unknown-row" id="row-${seg.id}">
                    <div class="unknown-row-top">
                        <span class="unknown-time">[${seg.elapsed || ''}]</span>
                        <button class="play-btn" data-session="${session.session_id}" data-seg="${seg.id}" onclick="playSegment(this.dataset.session, this.dataset.seg)">&#9654;</button>
                        <span class="speaker-badge" style="background:#f0f4ff;color:#667eea">${autoLabel}</span>
                        <span class="unknown-text">${escHtml(seg.text || '')}</span>
                    </div>
                    <div class="unknown-row-bottom">
                        <input class="name-input" id="input-${seg.id}" data-seg="${seg.id}" type="text" placeholder="Real name..." onkeydown="if(event.key==='Enter') assignSpeaker(this.dataset.seg)">
                        <button class="assign-btn" id="btn-${seg.id}" data-seg="${seg.id}" onclick="assignSpeaker(this.dataset.seg)">Assign</button>
                    </div></div>`;
            });

            html += '</div>';
            content.innerHTML = html;
        }

        function escHtml(s) {
            return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
        }

        function playSegment(sessionId, segId) {
            const audio = new Audio('/api/sessions/' + sessionId + '/audio/' + segId);
            audio.play();
        }

        function assignSpeaker(segId) {
            const input = document.getElementById('input-' + segId);
            const btn = document.getElementById('btn-' + segId);
            const name = (input.value || '').trim();
            if (!name || !currentSessionId) return;

            btn.disabled = true;
            btn.textContent = '…';

            fetch('/api/sessions/' + currentSessionId + '/annotate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ segment_id: segId, speaker_name: name })
            })
            .then(r => r.json())
            .then(() => {
                // Reload the full panel — batch rename may have updated multiple rows
                loadAnnotateTab(currentSessionId);
            })
            .catch(() => { btn.disabled = false; btn.textContent = 'Assign'; });
        }

        const recordBtn = document.getElementById('recordBtn');
        const status = document.getElementById('status');
        const diagnostics = document.getElementById('diagnostics');
        const liveTranscript = document.getElementById('liveTranscript');
        const sessionDot = document.getElementById('sessionDot');
        const sessionStatusText = document.getElementById('sessionStatusText');
        const disconnectionIndicator = document.getElementById('disconnectionIndicator');
        
        // Auto-scroll logic: track if user manually scrolled up
        liveTranscript.addEventListener('scroll', () => {
            const threshold = 50;
            const atBottom = (liveTranscript.scrollHeight - liveTranscript.scrollTop - liveTranscript.clientHeight) < threshold;
            userScrolledUp = !atBottom;
        });
        
        function autoScrollTranscript() {
            if (!userScrolledUp) {
                liveTranscript.scrollTop = liveTranscript.scrollHeight;
            }
        }
        
        function handleSessionStarted(message) {
            sessionDot.className = 'session-dot active';
            sessionStartTime = Date.now();
            sessionStatusText.textContent = 'Session active — 00:00:00';
            liveTranscript.innerHTML = '';
            userScrolledUp = false;
            // Update elapsed time every second
            if (sessionElapsedInterval) clearInterval(sessionElapsedInterval);
            sessionElapsedInterval = setInterval(() => {
                if (!sessionStartTime) return;
                const elapsed = Math.floor((Date.now() - sessionStartTime) / 1000);
                const h = Math.floor(elapsed / 3600);
                const m = Math.floor((elapsed % 3600) / 60);
                const s = elapsed % 60;
                const timeStr = [h, m, s].map(n => String(n).padStart(2, '0')).join(':');
                sessionStatusText.textContent = 'Session active \\u2014 ' + timeStr;
            }, 1000);
        }
        
        function handleTranscriptSegment(message) {
            const segmentDiv = document.createElement('div');
            segmentDiv.className = 'segment';
            
            // Parse timestamp to HH:MM:SS format
            let timeStr = '';
            if (message.timestamp) {
                const date = new Date(message.timestamp);
                timeStr = date.toTimeString().split(' ')[0]; // HH:MM:SS
            }
            
            const timeSpan = document.createElement('span');
            timeSpan.className = 'time';
            timeSpan.textContent = '[' + timeStr + ']';
            
            segmentDiv.appendChild(timeSpan);
            segmentDiv.appendChild(document.createTextNode(' ' + (message.text || '')));
            
            liveTranscript.appendChild(segmentDiv);
            autoScrollTranscript();
        }
        
        function handleSessionEnded(message) {
            sessionDot.className = 'session-dot ended';
            if (sessionElapsedInterval) {
                clearInterval(sessionElapsedInterval);
                sessionElapsedInterval = null;
            }
            sessionStartTime = null;
            const duration = message.duration_seconds || 0;
            const hours = Math.floor(duration / 3600);
            const minutes = Math.floor((duration % 3600) / 60);
            const seconds = Math.floor(duration % 60);
            const durationStr = [hours, minutes, seconds].map(n => String(n).padStart(2, '0')).join(':');
            const segCount = message.segment_count || 0;
            sessionStatusText.textContent = 'Session ended (' + durationStr + ', ' + segCount + ' segments)';

            // Show session file path if available
            if (message.session_file) {
                const endDiv = document.createElement('div');
                endDiv.className = 'segment';
                endDiv.style.color = '#999';
                endDiv.style.fontStyle = 'italic';
                endDiv.textContent = '\\u2014 Session ended. Transcript saved to: ' + message.session_file;
                liveTranscript.appendChild(endDiv);
                autoScrollTranscript();
            }

            // Load annotate tab with this session's unknown segments
            if (message.session_file) {
                // Derive session_id from file path (basename without extension)
                const parts = message.session_file.replace(/\\\\/g, '/').split('/');
                const fname = parts[parts.length - 1];
                const sid = fname.replace(/\\.txt$/, '');
                loadAnnotateTab(sid);
                // Show red dot on Annotate tab
                document.getElementById('annotateDot').style.display = 'block';
            }
        }
        
        function updateDiagnostics() {
            if (serverConnected && clientConnected) {
                diagnostics.className = 'diagnostics ok';
                diagnostics.innerHTML = '<h3><span class="status-dot green"></span>All systems connected</h3>' +
                    '<p>Double-tap Left Option (⌥) to start recording.</p>';
            } else if (serverConnected && !clientConnected) {
                diagnostics.className = 'diagnostics warning';
                diagnostics.innerHTML = '<h3><span class="status-dot yellow"></span>Client not connected</h3>' +
                    '<p>The voice client hasn\\'t connected yet. This could mean:</p>' +
                    '<ul>' +
                    '<li><strong>Input Monitoring</strong> not enabled — go to System Settings → Privacy & Security → Input Monitoring → enable your Terminal</li>' +
                    '<li><strong>Microphone</strong> not enabled — go to System Settings → Privacy & Security → Microphone → enable your Terminal</li>' +
                    '<li><strong>Accessibility</strong> not enabled (needed for auto-paste) — go to System Settings → Privacy & Security → Accessibility → enable your Terminal</li>' +
                    '<li>The client process crashed — check /tmp/voice-inject-client.log</li>' +
                    '</ul>' +
                    '<p style="margin-top:8px">After granting permissions, restart by pressing Ctrl+C and running <code>voice</code> again.</p>';
            } else {
                diagnostics.className = 'diagnostics error';
                diagnostics.innerHTML = '<h3><span class="status-dot red"></span>Disconnected from server</h3>' +
                    '<p>Reconnecting...</p>';
            }
        }
        
        // Connect to WebSocket
        function connectWebSocket() {
            ws = new WebSocket('ws://localhost:3000/ws');
            
            ws.onopen = () => {
                console.log('Connected to server');
                serverConnected = true;
                disconnectionIndicator.classList.remove('visible');
                updateDiagnostics();
            };
            
            ws.onmessage = (event) => {
                const message = JSON.parse(event.data);
                
                // Any message from the client means it's connected
                if (message.type === 'status' || message.type === 'transcript_segment') {
                    clientConnected = true;
                    lastClientMessage = Date.now();
                    updateDiagnostics();
                }
                
                if (message.type === 'warmup_started') {
                    recordBtn.disabled = true;
                    recordBtn.style.opacity = '0.4';
                    status.textContent = 'Warming up AI models…';
                } else if (message.type === 'warmup_complete') {
                    recordBtn.disabled = false;
                    recordBtn.style.opacity = '1';
                    status.textContent = 'Ready';
                } else if (message.type === 'status') {
                    if (message.mode === 'command') {
                        // Command mode status update
                        const cmdStatus = document.getElementById('commandStatus');
                        if (message.recording) {
                            cmdStatus.innerHTML = '<span class="status-dot" style="background:#f44336"></span> Recording — speak now...';
                        } else {
                            cmdStatus.innerHTML = '<span class="status-dot green"></span> Ready — waiting for hotkey';
                        }
                    } else {
                        // Meeting mode
                        isRecording = message.recording;
                        updateUI();
                    }
                } else if (message.type === 'session_started') {
                    handleSessionStarted(message);
                } else if (message.type === 'transcript_segment') {
                    handleTranscriptSegment(message);
                } else if (message.type === 'session_ended') {
                    handleSessionEnded(message);
                }
            };
            
            ws.onclose = () => {
                console.log('Disconnected, reconnecting in 2s...');
                serverConnected = false;
                clientConnected = false;
                disconnectionIndicator.classList.add('visible');
                updateDiagnostics();
                setTimeout(connectWebSocket, 2000);
            };
        }
        
        // Check if client is still alive (no messages in 30s = likely dead)
        setInterval(() => {
            if (serverConnected && lastClientMessage > 0 && (Date.now() - lastClientMessage) > 30000) {
                // Client was connected but went silent - might have crashed
                clientConnected = false;
                updateDiagnostics();
            }
        }, 10000);
        
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
        
        connectWebSocket();
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


# ---------------------------------------------------------------------------
# Speaker annotation endpoints
# ---------------------------------------------------------------------------

import re as _re

def _is_anonymous_speaker(label) -> bool:
    """Return True for auto-assigned diarization labels like 'Speaker 1'."""
    return label is None or bool(_re.match(r'^Speaker \d+$', str(label)))


@app.get("/api/sessions")
async def list_sessions():
    """List all sessions that have a .json file, newest first."""
    sessions = []
    for json_path in TRANSCRIPTS_DIR.glob("session_*.json"):
        try:
            with open(json_path) as f:
                data = json.load(f)
            known_count = sum(
                1 for seg in data.get("segments", [])
                if not _is_anonymous_speaker(seg.get("speaker"))
            )
            sessions.append({
                "session_id": data.get("session_id", json_path.stem),
                "started": data.get("started"),
                "ended": data.get("ended"),
                "segment_count": data.get("segment_count", len(data.get("segments", []))),
                "known_count": known_count,
            })
        except Exception as e:
            logger.warning(f"Could not read session file {json_path}: {e}")
    sessions.sort(key=lambda x: x.get("started") or "", reverse=True)
    return {"sessions": sessions}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Return the full session JSON."""
    json_path = TRANSCRIPTS_DIR / f"{session_id}.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    with open(json_path) as f:
        data = json.load(f)
    return data


@app.post("/api/sessions/{session_id}/annotate")
async def annotate_session(session_id: str, body: dict):
    """Annotate a segment with a speaker name and re-identify unlabelled segments."""
    segment_id = body.get("segment_id")
    speaker_name = body.get("speaker_name")
    if not segment_id or not speaker_name:
        raise HTTPException(status_code=400, detail="Both 'segment_id' and 'speaker_name' are required")

    json_path = TRANSCRIPTS_DIR / f"{session_id}.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    with open(json_path) as f:
        session = json.load(f)

    # Locate the annotated segment
    target_seg = next((s for s in session.get("segments", []) if s.get("id") == segment_id), None)
    if target_seg is None:
        raise HTTPException(status_code=404, detail=f"Segment '{segment_id}' not found in session '{session_id}'")

    # Load WAV helper
    def load_wav_float32(wav_path: Path):
        with wave.open(str(wav_path), 'r') as wf:
            frames = wf.readframes(wf.getnframes())
        return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    # Update the annotated segment and batch-rename all segments with the same auto-label
    old_speaker = target_seg.get("speaker")
    updated_segments = []
    for seg in session.get("segments", []):
        # Rename every segment that shares the same anonymous label (same diarization speaker)
        if old_speaker is not None and seg.get("speaker") == old_speaker:
            seg["speaker"] = speaker_name
            updated_segments.append({"id": seg["id"], "speaker": speaker_name})
    # Ensure the explicitly annotated segment is always updated
    if not any(u["id"] == segment_id for u in updated_segments):
        target_seg["speaker"] = speaker_name
        updated_segments.append({"id": segment_id, "speaker": speaker_name})

    # Extract embedding and re-identify remaining anonymous segments
    wav_path = TRANSCRIPTS_DIR / target_seg["audio_path"]
    embedding_available = False
    if wav_path.exists():
        try:
            identifier = get_speaker_identifier()
            db = get_speaker_db()
            audio = load_wav_float32(wav_path)
            embedding = identifier.get_embedding(audio, 16000)
            db.add_speaker(speaker_name, embedding)
            embedding_available = True
        except Exception as e:
            logger.warning(f"Speaker embedding unavailable ({e}); segment labelled without re-identification.")

    if embedding_available:
        for seg in session.get("segments", []):
            if not _is_anonymous_speaker(seg.get("speaker")):
                continue  # already has a real name
            seg_wav_path = TRANSCRIPTS_DIR / seg["audio_path"]
            if not seg_wav_path.exists():
                logger.warning(f"Audio file missing for segment {seg.get('id')}: {seg['audio_path']}")
                continue
            try:
                seg_audio = load_wav_float32(seg_wav_path)
                seg_embedding = identifier.get_embedding(seg_audio, 16000)
                name, similarity = db.find_closest(seg_embedding)
                if name is not None and similarity >= 0.85:
                    seg["speaker"] = name
                    updated_segments.append({"id": seg.get("id"), "speaker": name})
            except Exception as e:
                logger.warning(f"Could not identify speaker for segment {seg.get('id')}: {e}")

    # Persist updated session JSON
    with open(json_path, "w") as f:
        json.dump(session, f, indent=2)

    return {"updated_segments": updated_segments}


@app.get("/api/sessions/{session_id}/audio/{segment_id}")
async def get_segment_audio(session_id: str, segment_id: str):
    """Serve the WAV file for a given segment."""
    json_path = TRANSCRIPTS_DIR / f"{session_id}.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    with open(json_path) as f:
        session = json.load(f)

    seg = next((s for s in session.get("segments", []) if s.get("id") == segment_id), None)
    if seg is None:
        raise HTTPException(status_code=404, detail=f"Segment '{segment_id}' not found in session '{session_id}'")

    wav_path = TRANSCRIPTS_DIR / seg["audio_path"]
    if not wav_path.exists():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {seg['audio_path']}")

    return FileResponse(str(wav_path), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    print("🎙️  Voice Inject Server starting...")
    print("   API: http://localhost:3000")
    print("   UI: http://localhost:3000")
    print("   WebSocket: ws://localhost:3000/ws")
    print("   Endpoints:")
    print("     GET/POST /api/config - Settings")
    print("     GET /api/transcripts - Saved transcripts")
    print("     GET /health - Health check")
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=3000, log_level="info")

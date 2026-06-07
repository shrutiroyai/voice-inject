#!/usr/bin/env python3
"""Voice Inject Server — Full Version (Meeting + Command)."""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import yaml
import sys
import logging
from pathlib import Path
import json

def _load_dotenv():
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists(): return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ: os.environ[key] = value

import os
_load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)], force=True)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

CONFIG_DIR = Path.home() / ".voice-inject"
CONFIG_PATH = CONFIG_DIR / "config.yaml"
CONFIG_DIR.mkdir(exist_ok=True)

active_connections = []
warmup_state = {"type": "warmup_progress", "percent": 0, "message": "Starting..."}

def load_config():
    defaults = {"min_speech_energy": 180, "active_preset": "office"}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            data = yaml.safe_load(f) or {}
            defaults.update(data)
    return defaults

def save_config(config: dict):
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global warmup_state
    await websocket.accept()
    active_connections.append(websocket)
    await websocket.send_text(json.dumps(warmup_state))
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            # Broadcast to all other clients
            for connection in active_connections:
                if connection != websocket: await connection.send_text(data)
            
            # Persist warmup state
            if message.get("type") in ("warmup_started", "warmup_complete", "warmup_progress"):
                warmup_state = message
    except WebSocketDisconnect: active_connections.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in active_connections: active_connections.remove(websocket)

@app.get("/api/config")
async def get_config(): return load_config()

@app.post("/api/config")
async def update_config(new_config: dict):
    config = load_config()
    config.update(new_config)
    save_config(config)
    for connection in active_connections:
        await connection.send_text(json.dumps({"type": "config_updated", "config": config}))
    return {"success": True}

@app.get("/health")
async def health(): return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Voice Inject</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, sans-serif; background: #f5f5f7; display: flex; height: 100vh; }
        .sidebar { width: 300px; background: white; border-right: 1px solid #ddd; padding: 20px; display: flex; flex-direction: column; }
        .main { flex: 1; display: flex; flex-direction: column; padding: 40px; overflow-y: auto; }
        .card { background: white; border-radius: 12px; padding: 24px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 24px; }
        h1 { font-size: 24px; margin-bottom: 20px; }
        .status-badge { display: inline-flex; align-items: center; gap: 8px; padding: 6px 12px; border-radius: 20px; background: #eee; font-size: 13px; font-weight: 600; }
        .status-dot { width: 8px; height: 8px; border-radius: 50%; background: #ccc; }
        .status-dot.active { background: #34c759; }
        .status-dot.recording { background: #ff3b30; animation: pulse 1s infinite; }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
        .mode-section { margin-top: 30px; }
        .mode-card { display: flex; align-items: center; gap: 15px; padding: 15px; border-radius: 10px; border: 2px solid #eee; margin-bottom: 15px; }
        .mode-card.active { border-color: #667eea; background: #f0f4ff; }
        .recording-btn {
            width: 100%; padding: 15px; border-radius: 10px; border: none; background: #667eea; color: white;
            font-weight: 700; font-size: 16px; cursor: pointer; transition: all 0.3s; margin-top: 20px;
            display: flex; align-items: center; justify-content: center; gap: 10px;
        }
        .recording-btn.active {
            background: #ff3b30; box-shadow: 0 0 20px rgba(255, 59, 48, 0.4); animation: glow 1.5s infinite;
        }
        @keyframes glow {
            0% { box-shadow: 0 0 5px rgba(255, 59, 48, 0.4); }
            50% { box-shadow: 0 0 20px rgba(255, 59, 48, 0.7); }
            100% { box-shadow: 0 0 5px rgba(255, 59, 48, 0.4); }
        }
        .transcript-line { margin-bottom: 12px; line-height: 1.5; }
        .speaker { font-weight: 700; color: #667eea; margin-right: 8px; }
        .config-section { background: #f0f4ff; border-radius: 12px; padding: 20px; margin-bottom: 20px; }
        .preset-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-top: 10px; }
        .preset-btn { background: white; border: 2px solid #e0e0e0; border-radius: 8px; padding: 10px; cursor: pointer; font-size: 11px; font-weight: 600; display: flex; flex-direction: column; align-items: center; gap: 4px; }
        .preset-btn.active { border-color: #667eea; background: #667eea; color: white; }
        .progress-container { width: 100%; height: 6px; background: #eee; border-radius: 3px; margin-top: 15px; overflow: hidden; display: none; }
        .progress-bar { height: 100%; background: #667eea; width: 0%; transition: width 0.3s; }
    </style>
</head>
<body>
    <div class="sidebar">
        <h1>🎙️ Voice Inject</h1>
        <div class="status-badge">
            <div class="status-dot" id="statusDot"></div>
            <span id="statusText">Connecting...</span>
        </div>
        
        <div class="mode-section">
            <div class="mode-card" id="mode-command">
                <div style="font-weight:600">Command Mode</div>
                <div style="font-size:12px;color:#666">Double-tap Left Option</div>
            </div>
            <div class="mode-card" id="mode-meeting">
                <div style="font-weight:600">Meeting Mode</div>
                <div style="font-size:12px;color:#666">Click button to toggle</div>
            </div>
            
            <button id="meetingBtn" class="recording-btn" onclick="toggleMeeting()">
                <span id="meetingBtnIcon">▶️</span> <span id="meetingBtnText">Start Meeting</span>
            </button>
        </div>

        <div class="config-section" style="margin-top:auto">
            <h3 style="font-size:12px;margin-bottom:10px">SENSITIVITY</h3>
            <div class="preset-grid">
                <button class="preset-btn" id="p-laptop" onclick="setPreset('laptop',100)"><span>💻</span>Laptop</button>
                <button class="preset-btn" id="p-office" onclick="setPreset('office',180)"><span>🏢</span>Office</button>
                <button class="preset-btn" id="p-headphones" onclick="setPreset('headphones',350)"><span>🎧</span>Studio</button>
            </div>
        </div>
        
        <div id="hfSection" style="display:none; padding:15px; background:#fff3cd; border-radius:10px; font-size:12px">
            <b>HF Token Required</b><br>
            <input type="password" id="hfIn" placeholder="hf_..." style="width:100%; margin:8px 0; padding:5px">
            <button onclick="saveToken()" style="width:100%">Save Token</button>
        </div>
    </div>
    
    <div class="main">
        <div id="warmupBox">
            <div id="warmupMsg" style="color:#666;font-size:14px">Warming up models...</div>
            <div class="progress-container" id="pCont" style="display:block"><div class="progress-bar" id="pBar"></div></div>
        </div>
        
        <div class="card" style="flex:1; display:flex; flex-direction:column">
            <h3 style="margin-bottom:15px">Live Transcript</h3>
            <div id="transcript" style="flex:1; overflow-y:auto; font-size:15px">
                <p style="color:#999 italic">Meeting transcript will appear here...</p>
            </div>
        </div>
    </div>

    <script>
        let ws;
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        const transcript = document.getElementById('transcript');
        const pBar = document.getElementById('pBar');
        const warmupMsg = document.getElementById('warmupMsg');
        const hfIn = document.getElementById('hfIn');
        const hfSection = document.getElementById('hfSection');
        const meetingBtn = document.getElementById('meetingBtn');
        const meetingBtnText = document.getElementById('meetingBtnText');
        const meetingBtnIcon = document.getElementById('meetingBtnIcon');

        function toggleMeeting() {
            ws.send(JSON.stringify({type: 'toggle_meeting'}));
        }

        async function setPreset(id, e) {
            await fetch('/api/config', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({min_speech_energy:e, active_preset:id})});
            updateUI();
        }
        async function saveToken() {
            await fetch('/api/config', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({huggingface_token:hfIn.value.trim()})});
            hfSection.style.display = 'none';
        }
        async function updateUI() {
            const r = await fetch('/api/config'); const c = await r.json();
            document.querySelectorAll('.preset-btn').forEach(b=>b.classList.remove('active'));
            if(c.active_preset) {
                const btn = document.getElementById('p-'+c.active_preset);
                if (btn) btn.classList.add('active');
            }
            if(!c.huggingface_token) hfSection.style.display = 'block';
        }

        function connect() {
            ws = new WebSocket('ws://'+location.host+'/ws');
            ws.onopen = () => { statusText.innerText='Ready'; statusDot.className='status-dot active'; updateUI(); };
            ws.onmessage = (e) => {
                const m = JSON.parse(e.data);
                if(m.type==='warmup_progress'){ pBar.style.width=m.percent+'%'; warmupMsg.innerText=m.message; }
                if(m.type==='warmup_complete'){ document.getElementById('warmupBox').style.display='none'; }
                if(m.type==='transcript'){
                    const line = document.createElement('div'); line.className='transcript-line';
                    line.innerHTML = `<span class="speaker">${m.speaker}:</span><span>${m.text}</span>`;
                    transcript.appendChild(line); transcript.scrollTop = transcript.scrollHeight;
                }
                if(m.type==='status'){
                    if(m.recording){ statusText.innerText='Recording'; statusDot.className='status-dot recording'; }
                    else { statusText.innerText='Ready'; statusDot.className='status-dot active'; }
                    
                    document.querySelectorAll('.mode-card').forEach(c=>c.classList.remove('active'));
                    if(m.recording) {
                        const card = document.getElementById('mode-'+m.mode);
                        if (card) card.classList.add('active');
                    }
                    
                    if(m.mode === 'meeting') {
                        if(m.recording) {
                            meetingBtn.classList.add('active');
                            meetingBtnText.innerText = 'Stop Meeting';
                            meetingBtnIcon.innerText = '⏹️';
                        } else {
                            meetingBtn.classList.remove('active');
                            meetingBtnText.innerText = 'Start Meeting';
                            meetingBtnIcon.innerText = '▶️';
                        }
                    }
                }
            };
            ws.onclose = () => setTimeout(connect, 2000);
        }
        connect();
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000, log_level="info")

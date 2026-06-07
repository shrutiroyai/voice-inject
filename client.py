#!/usr/bin/env python3
"""Voice Inject Client — optimized for Meeting + Command Mode."""

import subprocess
import signal
import sys
import os
import logging
import sounddevice as sd
from pynput import keyboard
import numpy as np
import time
import asyncio
import websockets
import json
import threading
import queue
from pathlib import Path

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

_load_dotenv()

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1

# Global state
meeting_active = False        # Meeting mode: double-tap Right Option
command_recording = False     # Command mode: double-tap Left Option
command_buffer = []           # Audio buffer for command mode
last_option_r_press = 0
last_option_l_press = 0
DOUBLE_TAP_THRESHOLD = 0.6

# Message queue for WebSocket
message_queue = queue.Queue()
ws_connected = False
_warmup_done = False

_MLX_MODEL = "mlx-community/whisper-medium-mlx"
_LLM_MODEL = "mlx-community/Phi-3.5-mini-instruct-4bit"

_WHISPER_HALLUCINATIONS = {
    "thank you", "thank you.", "thanks.", "thanks for watching.",
    "thanks for watching", "thank you for watching.",
    "thank you for watching", "you", "bye.", "bye",
    "the end.", "the end", "subscribe.", "like and subscribe.",
    "more paste.", "more paste", "thanks for watching!", "thank you for watching!"
}

import yaml
from speaker_db import SpeakerDB

def get_config_setting(key, default):
    """Load a setting from ~/.voice-inject/config.yaml, falling back to ENV then default."""
    config_path = Path.home() / ".voice-inject" / "config.yaml"
    if config_path.exists():
        try:
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}
                if key in data:
                    return data[key]
        except Exception:
            pass
    return os.environ.get(key.upper(), default)

# --- AUDIO CUES ---
def play_cue(frequency=800, duration=0.1):
    """Play a short subtle sine-wave beep."""
    try:
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
        wave = 0.1 * np.sin(2 * np.pi * frequency * t)
        sd.play(wave, SAMPLE_RATE)
    except:
        pass

# === MLX WORKER THREAD ===

mlx_request_queue = queue.Queue()
_llm_model = None
_llm_tokenizer = None
_diarizer = None
_identifier = None
_speaker_db = None

def mlx_worker():
    """Dedicated thread for ALL AI operations to ensure stability on Apple Silicon."""
    global _llm_model, _llm_tokenizer, _warmup_done, _diarizer, _identifier, _speaker_db
    
    import mlx.core as mx
    import mlx_whisper
    from mlx_lm import load, generate
    from speaker_id import SpeakerDiarizer, SpeakerIdentifier
    
    mx.set_default_device(mx.gpu)
    _speaker_db = SpeakerDB()
    
    while True:
        request = mlx_request_queue.get()
        if request is None: break
        
        req_type = request.get("type")
        callback = request.get("callback")
        
        try:
            # Set HF token if available in config
            hf_token = get_config_setting("huggingface_token", "").strip()
            if hf_token:
                print(f"🔑 [Auth] Using HF Token: {hf_token[:6]}...{hf_token[-4:]}")
                os.environ["HF_TOKEN"] = hf_token
            else:
                print("⚠️ [Auth] No HF Token found in config.yaml")

            if req_type == "warmup":
                print("⏳ Warming up models...")
                message_queue.put({"type": "warmup_progress", "percent": 5, "message": "Starting warmup..."})
                
                # Whisper
                message_queue.put({"type": "warmup_progress", "percent": 15, "message": "Loading Whisper..."})
                silence = np.zeros(16000, dtype=np.float32)
                mlx_whisper.transcribe(silence, path_or_hf_repo=_MLX_MODEL, condition_on_previous_text=False)
                print("✅ Whisper warm")
                
                # Diarizer
                message_queue.put({"type": "warmup_progress", "percent": 35, "message": "Loading Diarizer..."})
                try:
                    _diarizer = SpeakerDiarizer(use_auth_token=hf_token)
                    print("✅ Diarizer warm")
                except Exception as e:
                    error_msg = str(e)
                    print(f"⚠️ Diarizer load failed: {error_msg}")
                    if "403" in error_msg or "gated" in error_msg.lower():
                        msg = "Access Denied: Accept terms at hf.co/pyannote/segmentation-3.0"
                    elif "401" in error_msg:
                        msg = "Invalid HF Token: Check your token in the UI/config"
                    else:
                        msg = f"Diarizer Error: {error_msg[:40]}"
                    message_queue.put({"type": "warmup_progress", "percent": 35, "message": msg})
                    time.sleep(5)
                
                # LLM
                message_queue.put({"type": "warmup_progress", "percent": 65, "message": "Loading LLM..."})
                if _llm_model is None:
                    _llm_model, _llm_tokenizer = load(_LLM_MODEL)
                print("✅ LLM warm")
                
                # Identifier
                message_queue.put({"type": "warmup_progress", "percent": 85, "message": "Loading Identifier..."})
                try:
                    _identifier = SpeakerIdentifier(_speaker_db, use_auth_token=hf_token)
                    print("✅ Identifier warm")
                except Exception as e:
                    error_msg = str(e)
                    print(f"⚠️ Identifier load failed: {error_msg}")
                    if "403" in error_msg or "gated" in error_msg.lower():
                        msg = "Access Denied: Accept terms at hf.co/pyannote/embedding"
                    else:
                        msg = f"Identifier Error: {error_msg[:40]}"
                    message_queue.put({"type": "warmup_progress", "percent": 85, "message": msg})
                    time.sleep(5)
                
                _warmup_done = True
                message_queue.put({"type": "warmup_complete"})
                print("🔥 Models ready\n")

            elif req_type == "transcribe":
                audio = request.get("audio")
                result = mlx_whisper.transcribe(audio, path_or_hf_repo=_MLX_MODEL, language="en", condition_on_previous_text=False)
                text = (result.get("text") or "").strip()
                if text.lower() in _WHISPER_HALLUCINATIONS: text = ""
                if callback: callback(text)

            elif req_type == "cleanup":
                raw_text = request.get("text")
                if not raw_text.strip():
                    if callback: callback(raw_text)
                    continue
                if _llm_model is None: _llm_model, _llm_tokenizer = load(_LLM_MODEL)
                
                prompt = f"""<|system|>
You are a speech-to-text post-processor. Your ONLY task is to fix grammar and punctuation.
Do NOT change the wording.
Do NOT change the tone.
Do NOT remove filler words.
Do NOT add any notes or comments.
Do NOT polish or rephrase the text.
Ensure there is a single space after every period, comma, or punctuation mark.
Output the cleaned version only.<|end|>
<|user|>
Fix grammar and punctuation for the following text. Keep all original words and tone:

{raw_text}<|end|>
<|assistant|>
"""
                response = generate(_llm_model, _llm_tokenizer, prompt=prompt, max_tokens=150)
                if "<|end|>" in response: response = response.split("<|end|>")[0]
                for stop in ["\n(", "\n\n", "\nNote:", "\n---"]:
                    if stop in response: response = response.split(stop)[0]
                if callback: callback(response.strip() or raw_text)

            elif req_type == "diarize":
                audio = request.get("audio")
                segments = _diarizer.get_speech_segments(audio)
                results = []
                for segment, _, _ in segments.itertracks(yield_label=True):
                    start, end = segment.start, segment.end
                    clip = audio[int(start*SAMPLE_RATE):int(end*SAMPLE_RATE)]
                    if len(clip) < 0.5 * SAMPLE_RATE: continue
                    
                    speaker = _identifier.identify(clip)
                    # Transcribe the segment
                    trans_res = mlx_whisper.transcribe(clip, path_or_hf_repo=_MLX_MODEL, language="en")
                    text = trans_res.get("text", "").strip()
                    if text:
                        results.append({"speaker": speaker, "text": text})
                if callback: callback(results)

        except Exception as e:
            print(f"⚠️ MLX Worker Error ({req_type}): {e}")
            if callback: callback(None)
        
        mlx_request_queue.task_done()

threading.Thread(target=mlx_worker, daemon=True).start()

# === MEETING MODE LOGIC ===

meeting_buffer = []

def meeting_loop():
    """Continuously process meeting audio in chunks."""
    global meeting_buffer
    while meeting_active:
        time.sleep(2.0)
        if len(meeting_buffer) < 2 * SAMPLE_RATE: continue
        
        captured = meeting_buffer
        meeting_buffer = []
        audio = np.concatenate(captured, axis=0).flatten()
        
        def handle_diarization(results):
            if results:
                for res in results:
                    msg = {"type": "transcript", "speaker": res["speaker"], "text": res["text"]}
                    message_queue.put(msg)
                    print(f"👥 {res['speaker']}: {res['text']}")

        mlx_request_queue.put({"type": "diarize", "audio": audio, "callback": handle_diarization})

# === AUDIO CALLBACK ===

def audio_callback(indata, frames, time_info, status):
    if meeting_active:
        meeting_buffer.append(indata.copy())
    if command_recording:
        command_buffer.append(indata.copy())

# === COMMAND MODE HELPERS ===

def paste_text(text: str):
    if not text: return
    if not text.endswith(" "): text += " "
    try:
        subprocess.run(["pbcopy"], input=text.encode(), check=True)
    except Exception as e:
        print(f"⚠️ Clipboard copy failed: {e}"); return
    subprocess.run(["osascript", "-e", 'tell application "System Events" to keystroke "v" using command down'], capture_output=True, text=True)

def handle_command_result(text: str):
    if text:
        def handle_cleanup(cleaned: str):
            if cleaned:
                print(f"✨ {cleaned}")
                paste_text(cleaned)
        mlx_request_queue.put({"type": "cleanup", "text": text, "callback": handle_cleanup})

def command_vad_loop():
    global command_buffer
    import webrtcvad
    vad = webrtcvad.Vad(3)
    frame_duration_ms = 30
    frame_size = int(SAMPLE_RATE * frame_duration_ms / 1000)
    silence_threshold_frames = int(0.3 * 1000 / frame_duration_ms)
    silence_frames = 0
    has_speech = False
    
    while command_recording:
        time.sleep(0.1)
        if not command_buffer: continue
        total_audio = np.concatenate(command_buffer, axis=0).flatten()
        if len(total_audio) < frame_size: continue
        last_frame = total_audio[-frame_size:]
        frame_bytes = last_frame.astype(np.int16).tobytes()
        try: is_speech = vad.is_speech(frame_bytes, SAMPLE_RATE)
        except: is_speech = True
        if is_speech: has_speech = True; silence_frames = 0
        else: silence_frames += 1
        if has_speech and silence_frames >= silence_threshold_frames:
            captured, command_buffer = command_buffer, []; silence_frames = 0; has_speech = False
            audio_data = np.concatenate(captured, axis=0)
            rms = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))
            if rms < int(get_config_setting("min_speech_energy", "180")): continue
            audio_float = audio_data.astype(np.float32).flatten() / 32768.0
            mlx_request_queue.put({"type": "transcribe", "audio": audio_float, "callback": handle_command_result})

def command_flush_remaining():
    global command_buffer
    captured, command_buffer = command_buffer, []
    if not captured: return
    audio_data = np.concatenate(captured, axis=0)
    rms = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))
    if rms < int(get_config_setting("min_speech_energy", "180")): return
    audio_float = audio_data.astype(np.float32).flatten() / 32768.0
    mlx_request_queue.put({"type": "transcribe", "audio": audio_float, "callback": handle_command_result})
    print("📋 Command mode done.\n")

# === HOTKEYS ===

def toggle_meeting():
    global meeting_active, meeting_buffer
    if not meeting_active:
        meeting_active = True
        meeting_buffer = []
        play_cue(frequency=1000) # High blip for START
        print("📁 Meeting Mode: STARTED")
        message_queue.put({"type": "status", "recording": True, "mode": "meeting"})
        threading.Thread(target=meeting_loop, daemon=True).start()
    else:
        meeting_active = False
        play_cue(frequency=600) # Low blip for STOP
        print("📁 Meeting Mode: STOPPED")
        message_queue.put({"type": "status", "recording": False, "mode": "meeting"})

def toggle_command():
    global command_recording, command_buffer
    if not command_recording:
        command_recording = True
        command_buffer = []
        play_cue(frequency=1000) # High blip for START
        print("🎤 Command Mode: STARTED")
        message_queue.put({"type": "status", "recording": True, "mode": "command"})
        threading.Thread(target=command_vad_loop, daemon=True).start()
    else:
        command_recording = False
        play_cue(frequency=600) # Low blip for STOP
        print("⏹️ Command Mode: STOPPING")
        message_queue.put({"type": "status", "recording": False, "mode": "command"})
        threading.Thread(target=command_flush_remaining, daemon=True).start()

def on_press(key):
    global last_option_r_press, last_option_l_press
    if key == keyboard.Key.alt_r:
        now = time.time()
        if (now - last_option_r_press) < DOUBLE_TAP_THRESHOLD:
            toggle_meeting(); last_option_r_press = 0
        else: last_option_r_press = now
    elif key == keyboard.Key.alt_l:
        now = time.time()
        if (now - last_option_l_press) < DOUBLE_TAP_THRESHOLD:
            toggle_command(); last_option_l_press = 0
        else: last_option_l_press = now

# === WS CLIENT ===

async def websocket_client():
    global ws_connected
    uri = "ws://localhost:3000/ws"
    while True:
        try:
            async with websockets.connect(uri) as websocket:
                ws_connected = True; print("✅ Connected to server")
                if _warmup_done: message_queue.put({"type": "warmup_complete"})
                else: message_queue.put({"type": "warmup_progress", "percent": 0, "message": "Waiting for worker..."})
                async def send():
                    while True:
                        while not message_queue.empty():
                            await websocket.send(json.dumps(message_queue.get_nowait()))
                        await asyncio.sleep(0.1)
                async def recv():
                    async for message in websocket:
                        try:
                            msg = json.loads(message)
                            if msg.get("type") == "toggle_meeting":
                                toggle_meeting()
                        except:
                            pass
                await asyncio.gather(send(), recv())
        except: ws_connected = False; await asyncio.sleep(5)

def main():
    print("🎙️ Voice Inject (Full Mode)")
    print("   Double-tap Right Option (⌥) → Meeting")
    print("   Double-tap Left Option  (⌥) → Command")
    threading.Thread(target=lambda: asyncio.run(websocket_client()), daemon=True).start()
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    mlx_request_queue.put({"type": "warmup"})
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16", callback=audio_callback):
        with keyboard.Listener(on_press=on_press) as listener: listener.join()

if __name__ == "__main__":
    main()

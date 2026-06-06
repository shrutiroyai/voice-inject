#!/usr/bin/env python3
"""Voice Inject Client — faster-whisper with toggle recording."""

import subprocess
import tempfile
import wave
import sounddevice as sd
from pynput import keyboard
import numpy as np
import time
import asyncio
import websockets
import json
import threading
import queue

SAMPLE_RATE = 16000
CHANNELS = 1
DEBUG_SAVE_AUDIO = True

# Global state
audio_buffer = []
is_recording = False
whisper_model = None
last_option_press = 0
DOUBLE_TAP_THRESHOLD = 0.4

# Message queue for WebSocket
message_queue = queue.Queue()
ws_connected = False


def load_whisper():
    """Lazy load faster-whisper model."""
    global whisper_model
    if whisper_model is None:
        print("  📦 Loading faster-whisper 'small' model (first time only)...")
        from faster_whisper import WhisperModel
        whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
        print("  ✅ Model loaded with CTranslate2 optimization")
    return whisper_model


def audio_callback(indata, frames, time_info, status):
    """Callback for audio recording."""
    if is_recording:
        audio_buffer.append(indata.copy())


def paste_text(text: str):
    """Copy to clipboard and attempt auto-paste."""
    try:
        subprocess.run(["pbcopy"], input=text.encode(), check=True)
    except Exception as e:
        print(f"⚠️  Clipboard copy failed: {e}")
        return
    
    result = subprocess.run([
        "osascript", "-e",
        'tell application "System Events" to keystroke "v" using command down'
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("⚠️  Auto-paste failed (use Cmd+V manually). Text is in clipboard.")


def transcribe_and_process():
    """Transcribe audio with faster-whisper."""
    global audio_buffer
    
    if not audio_buffer:
        print("⚠️  No audio recorded.")
        return
    
    audio_data = np.concatenate(audio_buffer, axis=0)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
        with wave.open(temp_wav.name, 'wb') as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(audio_data.tobytes())
        
        temp_path = temp_wav.name
    
    try:
        print("  🎤 Transcribing with faster-whisper...")
        start_time = time.time()
        
        model = load_whisper()
        segments, info = model.transcribe(temp_path, language="en", beam_size=5)
        
        raw_text = " ".join([segment.text.strip() for segment in segments])
        elapsed = time.time() - start_time
        print(f"  ⚡ Transcribed in {elapsed:.2f}s")
        
        if not raw_text:
            print("⚠️  No speech detected.")
            return
        
        print(f"🎤 Raw: {raw_text}")
        
        # Basic cleanup
        cleaned_text = raw_text.strip()
        if cleaned_text and cleaned_text[0].islower():
            cleaned_text = cleaned_text[0].upper() + cleaned_text[1:]
        if cleaned_text and cleaned_text[-1] not in '.!?':
            cleaned_text += '.'
        
        print(f"✨ Clean: {cleaned_text}")
        
        # Send to UI via WebSocket
        message_queue.put({
            "type": "transcript",
            "text": cleaned_text
        })
        
        # Auto-paste
        paste_text(cleaned_text)
        print("📋 Pasted!\n")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        import os
        if DEBUG_SAVE_AUDIO:
            import shutil
            debug_path = f"/tmp/voice-inject-debug-{int(time.time())}.wav"
            try:
                shutil.copy(temp_path, debug_path)
                print(f"🔍 DEBUG: Audio saved to {debug_path}")
            except:
                pass
        
        try:
            os.unlink(temp_path)
        except:
            pass


def toggle_recording():
    """Toggle recording on/off."""
    global is_recording, audio_buffer
    
    if not is_recording:
        is_recording = True
        audio_buffer = []
        print("🔴 Recording...")
        # Send status to UI
        message_queue.put({
            "type": "status",
            "recording": True
        })
    else:
        is_recording = False
        print("⏹️  Processing...")
        # Send status to UI
        message_queue.put({
            "type": "status",
            "recording": False
        })
        # Process in background thread to avoid blocking
        threading.Thread(target=transcribe_and_process, daemon=True).start()


def on_press(key):
    """Handle key press events for Right Option double-tap."""
    global last_option_press
    
    if key == keyboard.Key.alt_r:
        current_time = time.time()
        time_since_last = current_time - last_option_press
        
        if time_since_last < DOUBLE_TAP_THRESHOLD:
            toggle_recording()
            last_option_press = 0
        else:
            last_option_press = current_time


def on_release(key):
    """Handle key release events."""
    if key == keyboard.Key.esc:
        return False


async def websocket_client():
    """WebSocket client that runs in background thread."""
    global ws_connected
    uri = "ws://localhost:3000/ws"
    
    while True:
        try:
            async with websockets.connect(uri) as websocket:
                ws_connected = True
                print("✅ Connected to UI\n")
                
                # Create tasks for sending and receiving
                async def send_messages():
                    while True:
                        try:
                            # Check queue for messages to send
                            while not message_queue.empty():
                                message = message_queue.get_nowait()
                                print(f"📤 Sending to UI: {message}")
                                await websocket.send(json.dumps(message))
                            await asyncio.sleep(0.1)
                        except Exception as e:
                            print(f"Send error: {e}")
                            break
                
                async def receive_messages():
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            if data.get("type") == "toggle_recording":
                                # UI requested toggle
                                toggle_recording()
                        except Exception as e:
                            print(f"Receive error: {e}")
                
                # Run both tasks concurrently
                await asyncio.gather(
                    send_messages(),
                    receive_messages(),
                    return_exceptions=True
                )
        
        except Exception as e:
            ws_connected = False
            print(f"⚠️  WebSocket disconnected, retrying in 5s...")
            await asyncio.sleep(5)


def start_websocket_thread():
    """Start WebSocket in background thread with its own event loop."""
    def run_websocket():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(websocket_client())
    
    thread = threading.Thread(target=run_websocket, daemon=True)
    thread.start()


def main():
    """Main entry point."""
    print("🎙️  Voice Inject Client — faster-whisper edition")
    print("   Using: faster-whisper 'small' (int8)")
    print("   Trigger: Double-tap Right Option (⌥) to toggle recording")
    print("   UI: http://localhost:3000")
    print("   Press Esc to quit.\n")
    
    # Pre-load Whisper model
    load_whisper()
    print("✅ Whisper model loaded\n")
    
    # Start WebSocket connection in background
    start_websocket_thread()
    
    # Start audio input stream
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype="int16", callback=audio_callback):
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()


if __name__ == "__main__":
    main()

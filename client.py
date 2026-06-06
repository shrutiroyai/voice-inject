#!/usr/bin/env python3
"""Voice Inject Client — faster-whisper with toggle recording."""

import subprocess
import tempfile
import wave
import signal
import sys
import sounddevice as sd
from pynput import keyboard
import numpy as np
import time
import asyncio
import websockets
import json
import threading
import queue
from datetime import datetime
from pathlib import Path

SAMPLE_RATE = 16000
CHANNELS = 1
DEBUG_SAVE_AUDIO = True

# Global state
audio_buffer = []
is_recording = False
whisper_model = None
last_option_press = 0
DOUBLE_TAP_THRESHOLD = 0.4
continuous_transcriber = None

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
    """Callback for audio recording — feeds both continuous and dictation buffers."""
    # Always feed the continuous transcriber
    if continuous_transcriber is not None:
        continuous_transcriber.feed(indata.copy())
    # Feed dictation buffer only when quick-dictation is active
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
    """Handle key release events. No-op; exit via SIGINT (Ctrl+C)."""
    pass


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


class ContinuousTranscriber:
    """Background thread that transcribes audio in 10-second batches."""

    BATCH_INTERVAL = 10  # seconds
    TRANSCRIPTS_DIR = Path("transcripts")

    def __init__(self, model, sample_rate: int, message_queue: queue.Queue):
        self._model = model
        self._sample_rate = sample_rate
        self._message_queue = message_queue
        self._buffer: list = []
        self._buffer_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._segment_count: int = 0
        self._session_start_time: datetime | None = None
        self._file_handle = None
        self.session_file_path: Path | None = None

    def start(self) -> None:
        """Start continuous transcription. Creates session file, writes header."""
        self._stop_event.clear()
        self._session_start_time = datetime.now()

        # Create transcripts/ directory if needed
        self.TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

        # Generate session file path
        timestamp_str = self._session_start_time.strftime("%Y-%m-%d_%H-%M-%S")
        self.session_file_path = self.TRANSCRIPTS_DIR / f"session_{timestamp_str}.txt"

        # Open the file and write header
        self._file_handle = open(self.session_file_path, "w", encoding="utf-8")
        self._write_header()

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print("🟢 Continuous transcription started")

    def stop(self) -> None:
        """Stop transcription. Transcribes remainder ≥1s, writes footer, sends session_ended."""
        # 1. Signal the stop event
        self._stop_event.set()

        # 2. Join the thread
        if self._thread is not None:
            self._thread.join(timeout=15)
            self._thread = None

        # 3. Get remaining buffer under lock
        with self._buffer_lock:
            remaining = self._buffer
            self._buffer = []

        # 4/5. Transcribe remainder if ≥ 16000 samples (1 second at 16kHz), discard if < 16000
        if remaining:
            audio_data = np.concatenate(remaining, axis=0)
            total_samples = audio_data.shape[0] if audio_data.ndim == 1 else audio_data.shape[0]

            if total_samples >= 16000:
                # 6. Handle transcription failure during shutdown
                try:
                    # Convert int16 samples to float32 normalized to [-1, 1]
                    audio_float = audio_data.astype(np.float32).flatten() / 32768.0
                    segments, info = self._model.transcribe(
                        audio_float, language="en", beam_size=5
                    )
                    text = " ".join([segment.text.strip() for segment in segments]).strip()
                    if text:
                        self._segment_count += 1
                        self._write_segment(text)
                except Exception as e:
                    print(f"⚠️  Transcription failed during shutdown: {e}")
                    # Append untranscribed note to session file
                    if self._file_handle and not self._file_handle.closed:
                        try:
                            self._file_handle.write("[UNTRANSCRIBED] Final audio segment could not be processed\n")
                            self._file_handle.flush()
                        except Exception:
                            pass
            # else: buffer < 16000 samples, discard silently

        # 7. Write footer
        if self._file_handle and not self._file_handle.closed:
            self._write_footer()

        # 8. Close file
        if self._file_handle and not self._file_handle.closed:
            self._file_handle.close()
            self._file_handle = None

        # 9. Send session_ended message to the message queue
        if self._session_start_time is not None:
            duration_seconds = (datetime.now() - self._session_start_time).total_seconds()
            self._message_queue.put({
                "type": "session_ended",
                "duration_seconds": duration_seconds,
                "segment_count": self._segment_count
            })

        print("🔴 Continuous transcription stopped")

    def feed(self, audio_chunk: np.ndarray) -> None:
        """Called from audio_callback to feed samples into the continuous buffer."""
        with self._buffer_lock:
            self._buffer.append(audio_chunk)

    def _run_loop(self) -> None:
        """Timer loop: sleep for BATCH_INTERVAL, grab buffer, transcribe, dispatch."""
        while not self._stop_event.is_set():
            # Wait for the batch interval or until stop is signaled
            if self._stop_event.wait(timeout=self.BATCH_INTERVAL):
                break  # Stop was signaled

            # Atomically swap buffer with a fresh list
            with self._buffer_lock:
                captured = self._buffer
                self._buffer = []

            if not captured:
                continue

            # Concatenate all chunks into a single array
            audio_data = np.concatenate(captured, axis=0)

            # Transcribe the buffer
            text = self._transcribe_buffer(audio_data)

            if not text:
                continue

            self._segment_count += 1
            print(f"📝 [{self._segment_count}] {text}")

            # Write segment to session file
            self._write_segment(text)

            # Put the transcript segment in the message queue
            self._message_queue.put({
                "type": "transcript_segment",
                "text": text,
                "timestamp": datetime.now().isoformat()
            })

    def _write_header(self) -> None:
        """Write session file header with start timestamp and separator."""
        try:
            start_str = self._session_start_time.strftime("%Y-%m-%d %H:%M:%S")
            self._file_handle.write("# Voice Inject — Live Transcription Session\n")
            self._file_handle.write(f"# Started: {start_str}\n")
            self._file_handle.write("-" * 40 + "\n")
            self._file_handle.flush()
        except Exception as e:
            print(f"⚠️  Session file header write failed: {e}")

    def _write_segment(self, text: str) -> None:
        """Append segment to session file with elapsed timestamp prefix and flush."""
        try:
            if self._file_handle and not self._file_handle.closed:
                elapsed = datetime.now() - self._session_start_time
                total_seconds = int(elapsed.total_seconds())
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                seconds = total_seconds % 60
                timestamp = f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"
                self._file_handle.write(f"{timestamp} {text}\n")
                self._file_handle.flush()
        except Exception as e:
            print(f"⚠️  Session file write failed: {e}")

    def _write_footer(self) -> None:
        """Write session file footer with separator, end timestamp, duration, and segment count."""
        try:
            end_time = datetime.now()
            end_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
            elapsed = end_time - self._session_start_time
            total_seconds = int(elapsed.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            self._file_handle.write("-" * 40 + "\n")
            self._file_handle.write(f"# Session ended: {end_str}\n")
            self._file_handle.write(f"# Duration: {duration_str}\n")
            self._file_handle.write(f"# Segments: {self._segment_count}\n")
            self._file_handle.flush()
        except Exception as e:
            print(f"⚠️  Session file footer write failed: {e}")

    def _transcribe_buffer(self, audio_data: np.ndarray) -> str | None:
        """Transcribe audio via faster-whisper. Returns text or None on empty/error."""
        try:
            # Convert int16 samples to float32 normalized to [-1, 1] for faster-whisper
            audio_float = audio_data.astype(np.float32).flatten() / 32768.0

            segments, info = self._model.transcribe(
                audio_float, language="en", beam_size=5
            )

            text = " ".join([segment.text.strip() for segment in segments]).strip()

            if not text:
                return None

            return text

        except Exception as e:
            print(f"⚠️  Transcription error: {e}")
            return None


def main():
    """Main entry point."""
    global continuous_transcriber

    print("🎙️  Voice Inject Client — faster-whisper edition")
    print("   Using: faster-whisper 'small' (int8)")
    print("   Trigger: Double-tap Right Option (⌥) to toggle recording")
    print("   UI: http://localhost:3000")
    print("   Press Ctrl+C to quit.\n")
    
    # Pre-load Whisper model
    load_whisper()
    print("✅ Whisper model loaded\n")
    
    # Start WebSocket connection in background
    start_websocket_thread()

    # Register SIGINT handler for graceful shutdown
    def sigint_handler(signum, frame):
        print("\n⏹️  Ctrl+C received, shutting down gracefully...")
        if continuous_transcriber is not None:
            continuous_transcriber.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)

    # Start continuous transcription
    continuous_transcriber = ContinuousTranscriber(
        model=whisper_model,
        sample_rate=SAMPLE_RATE,
        message_queue=message_queue,
    )
    continuous_transcriber.start()

    # Send session_started message via message queue
    message_queue.put({
        "type": "session_started",
        "session_file": str(continuous_transcriber.session_file_path),
    })

    print("🎙️  Live transcription active (10-second batches)")
    print(f"   Session file: {continuous_transcriber.session_file_path}\n")

    # Start audio input stream
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype="int16", callback=audio_callback):
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()


if __name__ == "__main__":
    main()

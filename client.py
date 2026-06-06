#!/usr/bin/env python3
"""Voice Inject Client — faster-whisper with VAD-based live transcription."""

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

# Global state
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
    """Callback for audio recording — only feeds transcriber when recording is active."""
    if is_recording and continuous_transcriber is not None:
        continuous_transcriber.feed(indata.copy())


def toggle_recording():
    """Toggle recording on/off. This is the master control for live transcription."""
    global is_recording
    
    if not is_recording:
        is_recording = True
        print("🔴 Recording — live transcription active")
        message_queue.put({
            "type": "status",
            "recording": True
        })
    else:
        is_recording = False
        # Clear the buffer so nothing stale gets transcribed
        if continuous_transcriber is not None:
            continuous_transcriber.toggle_pause()
            continuous_transcriber.toggle_pause()  # This clears the buffer
        print("⏹️  Paused — transcription stopped")
        message_queue.put({
            "type": "status",
            "recording": False
        })


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
    """Background transcriber using Voice Activity Detection (VAD) to find natural sentence boundaries."""

    SILENCE_THRESHOLD = 0.8  # seconds of silence before cutting a segment
    MAX_SEGMENT_DURATION = 30  # max seconds before forcing a cut
    MIN_SPEECH_ENERGY = 500  # minimum RMS energy to consider as actual speech (filters noise)
    TRANSCRIPT_FILE = Path("transcripts") / "transcript.txt"

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
        self._silence_frames: int = 0
        self._has_speech: bool = False
        self._is_paused: bool = False
        # VAD setup
        import webrtcvad
        self._vad = webrtcvad.Vad(3)  # Aggressiveness 3 = most aggressive filtering (less false positives)
        # How many consecutive silent frames = silence threshold
        # Each frame is 30ms at 16kHz (480 samples)
        self._frame_duration_ms = 30
        self._frame_size = int(self._sample_rate * self._frame_duration_ms / 1000)  # 480 samples
        self._silence_frames_threshold = int(self.SILENCE_THRESHOLD * 1000 / self._frame_duration_ms)  # ~26 frames
        self._max_frames = int(self.MAX_SEGMENT_DURATION * 1000 / self._frame_duration_ms)

    def start(self) -> None:
        """Start continuous transcription with VAD."""
        self._stop_event.clear()
        self._session_start_time = datetime.now()
        self._silence_frames = 0
        self._has_speech = False

        # Create transcripts/ directory if needed
        self.TRANSCRIPT_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Append a session separator to the single transcript file
        with open(self.TRANSCRIPT_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n--- Session: {self._session_start_time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print("🟢 Live transcription started (VAD-based)")

    def stop(self) -> None:
        """Stop transcription. Transcribes any remaining audio."""
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=15)
            self._thread = None

        # Transcribe any remaining buffer
        with self._buffer_lock:
            remaining = self._buffer
            self._buffer = []

        if remaining:
            audio_data = np.concatenate(remaining, axis=0)
            total_samples = audio_data.shape[0] if audio_data.ndim == 1 else audio_data.shape[0]
            if total_samples >= self._sample_rate:  # at least 1 second
                text = self._transcribe_buffer(audio_data)
                if text:
                    self._segment_count += 1
                    self._write_segment(text)
                    self._message_queue.put({
                        "type": "transcript_segment",
                        "text": text,
                        "timestamp": datetime.now().isoformat()
                    })

        # Send session_ended
        if self._session_start_time:
            duration = (datetime.now() - self._session_start_time).total_seconds()
            self._message_queue.put({
                "type": "session_ended",
                "duration_seconds": duration,
                "segment_count": self._segment_count
            })

        print("🔴 Live transcription stopped")

    def feed(self, audio_chunk: np.ndarray) -> None:
        """Called from audio_callback to feed samples. Skipped when paused."""
        if self._is_paused:
            return
        with self._buffer_lock:
            self._buffer.append(audio_chunk)

    def toggle_pause(self) -> bool:
        """Toggle pause state. Returns True if now paused, False if resumed."""
        self._is_paused = not self._is_paused
        if self._is_paused:
            # Clear buffer when pausing to avoid transcribing stale audio
            with self._buffer_lock:
                self._buffer = []
            self._silence_frames = 0
            self._has_speech = False
        return self._is_paused

    def _run_loop(self) -> None:
        """VAD loop: monitors buffer for speech segments ending in silence."""
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=0.1)  # Check every 100ms

            with self._buffer_lock:
                if not self._buffer:
                    continue
                # Check the latest audio for VAD
                current_buffer = list(self._buffer)

            # Get total samples in buffer
            total_audio = np.concatenate(current_buffer, axis=0).flatten()
            total_frames = len(total_audio) // self._frame_size

            if total_frames == 0:
                continue

            # Check the last few frames for silence/speech
            last_frame_start = max(0, len(total_audio) - self._frame_size)
            last_frame = total_audio[last_frame_start:last_frame_start + self._frame_size]

            if len(last_frame) == self._frame_size:
                # Convert to bytes for webrtcvad (expects 16-bit PCM bytes)
                frame_bytes = last_frame.astype(np.int16).tobytes()
                try:
                    is_speech = self._vad.is_speech(frame_bytes, self._sample_rate)
                except Exception:
                    is_speech = True  # Assume speech on error

                if is_speech:
                    self._has_speech = True
                    self._silence_frames = 0
                else:
                    self._silence_frames += 1

            # Determine if we should cut and transcribe
            total_duration_s = len(total_audio) / self._sample_rate
            should_cut = False

            if self._has_speech and self._silence_frames >= self._silence_frames_threshold:
                # Natural silence boundary detected
                should_cut = True
            elif total_duration_s >= self.MAX_SEGMENT_DURATION:
                # Force cut at max duration to avoid memory buildup
                should_cut = True

            if should_cut and total_duration_s >= 0.5:  # Don't transcribe tiny fragments
                # Grab and clear the buffer
                with self._buffer_lock:
                    captured = self._buffer
                    self._buffer = []

                audio_data = np.concatenate(captured, axis=0)
                self._silence_frames = 0
                self._has_speech = False

                # Check if audio has enough energy to be actual speech (not just noise)
                rms = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))
                if rms < self.MIN_SPEECH_ENERGY:
                    continue  # Skip — likely just background noise

                # Transcribe in this thread (it's already a background thread)
                text = self._transcribe_buffer(audio_data)
                if text:
                    self._segment_count += 1
                    print(f"📝 [{self._segment_count}] {text}")
                    self._write_segment(text)
                    self._message_queue.put({
                        "type": "transcript_segment",
                        "text": text,
                        "timestamp": datetime.now().isoformat()
                    })

    def _write_segment(self, text: str) -> None:
        """Append segment to the single transcript file."""
        try:
            elapsed = datetime.now() - self._session_start_time
            total_seconds = int(elapsed.total_seconds())
            h = total_seconds // 3600
            m = (total_seconds % 3600) // 60
            s = total_seconds % 60
            timestamp = f"[{h:02d}:{m:02d}:{s:02d}]"
            with open(self.TRANSCRIPT_FILE, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} {text}\n")
        except Exception as e:
            print(f"⚠️  Transcript write failed: {e}")

    def _transcribe_buffer(self, audio_data: np.ndarray) -> str | None:
        """Transcribe audio via faster-whisper. Returns text or None."""
        try:
            audio_float = audio_data.astype(np.float32).flatten() / 32768.0
            segments, info = self._model.transcribe(audio_float, language="en", beam_size=5)
            text = " ".join([seg.text.strip() for seg in segments]).strip()
            return text if text else None
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

    # Initialize continuous transcriber (starts paused — user must press record)
    continuous_transcriber = ContinuousTranscriber(
        model=whisper_model,
        sample_rate=SAMPLE_RATE,
        message_queue=message_queue,
    )
    continuous_transcriber.start()

    # Send session_started message via message queue
    message_queue.put({
        "type": "session_started",
        "session_file": str(continuous_transcriber.TRANSCRIPT_FILE),
    })

    print("🎙️  Ready — press Record button or double-tap Right Option (⌥) to start")
    print(f"   Transcript: {continuous_transcriber.TRANSCRIPT_FILE}\n")

    # Start audio input stream
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype="int16", callback=audio_callback):
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()


if __name__ == "__main__":
    main()

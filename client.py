#!/usr/bin/env python3
"""Voice Inject Client — two modes: Meeting Transcription + Command (auto-paste)."""

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
import wave
import re
from datetime import datetime
from pathlib import Path
from scipy.spatial.distance import cdist
from speaker_id import SpeakerIdentifier, SpeakerDiarizer
from speaker_db import SpeakerDB


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
meeting_active = False       # Meeting mode: record button in browser
command_recording = False     # Command mode: double-tap Left Option
command_buffer = []           # Audio buffer for command mode
last_option_press = 0
DOUBLE_TAP_THRESHOLD = 0.6
continuous_transcriber = None
speaker_db = SpeakerDB()
speaker_identifier = SpeakerIdentifier()
speaker_diarizer = SpeakerDiarizer()

# Message queue for WebSocket
message_queue = queue.Queue()
ws_connected = False
_warmup_done = False  # track so reconnects can re-broadcast warmup state


_MLX_MODEL = "mlx-community/whisper-small-mlx"
_mlx_lock = threading.Lock()  # mlx_whisper is not thread-safe


def transcribe_audio(audio_float: np.ndarray) -> str:
    """Transcribe float32 audio via mlx-whisper (M2 Neural Engine). Returns text or ''."""
    import mlx_whisper
    with _mlx_lock:
        result = mlx_whisper.transcribe(
            audio_float,
            path_or_hf_repo=_MLX_MODEL,
            condition_on_previous_text=False,
        )
    return (result.get("text") or "").strip()


def _warmup_models():
    """Load ML models into memory at startup so first use has no lag."""
    import warnings
    warnings.filterwarnings("ignore")

    message_queue.put({"type": "warmup_started"})
    print("⏳ Warming up models...")

    try:
        # Warm mlx_whisper
        import mlx_whisper
        silence = np.zeros(16000, dtype=np.float32)
        with _mlx_lock:
            mlx_whisper.transcribe(silence, path_or_hf_repo=_MLX_MODEL,
                                   condition_on_previous_text=False)
        print("✅ Whisper warm")
    except Exception as e:
        print(f"⚠️  Whisper warm-up failed: {e}")

    try:
        # Warm diarizer
        speaker_diarizer._ensure_pipeline()
        print("✅ Diarizer warm")
    except Exception as e:
        print(f"⚠️  Diarizer warm-up failed: {e}")

    global _warmup_done
    _warmup_done = True
    message_queue.put({"type": "warmup_complete"})
    print("🔥 Models ready\n")


def audio_callback(indata, frames, time_info, status):
    """Route audio to the active mode's buffer."""
    # Meeting mode: feed continuous transcriber
    if meeting_active and continuous_transcriber is not None:
        continuous_transcriber.feed(indata.copy())
    # Command mode: accumulate for paste (hard cap at 60 s to prevent runaway recordings)
    if command_recording:
        command_buffer.append(indata.copy())
        if len(command_buffer) * frames / SAMPLE_RATE > 60:
            toggle_command()


# === COMMAND MODE (double-tap Left Option → transcribe → paste) ===

def paste_text(text: str):
    """Copy to clipboard and auto-paste via Cmd+V."""
    try:
        subprocess.run(["pbcopy"], input=text.encode(), check=True)
    except Exception as e:
        print(f"⚠️  Clipboard copy failed: {e}")
        return
    subprocess.run([
        "osascript", "-e",
        'tell application "System Events" to keystroke "v" using command down'
    ], capture_output=True, text=True)


def command_transcribe_and_paste():
    """Transcribe command buffer and paste result."""
    global command_buffer
    # Atomically drain the buffer so concurrent calls don't transcribe the same audio
    captured, command_buffer = command_buffer, []
    if not captured:
        print("⚠️  No audio recorded.")
        return

    try:
        audio_float = np.concatenate(captured, axis=0).astype(np.float32).flatten() / 32768.0
        text = transcribe_audio(audio_float)

        if not text:
            print("⚠️  No speech detected.")
            return

        # Speaker identification — skip if no speakers enrolled (avoids pyannote inference cost)
        identified = False
        speaker_name = None
        if speaker_db.list_speakers():
            try:
                speaker_name = speaker_identifier.identify(audio_float, SAMPLE_RATE, speaker_db)
                identified = bool(speaker_name) and speaker_name != "Unknown"
            except Exception as sid_err:
                logger.debug("Speaker ID unavailable: %s", sid_err)

        # Basic cleanup
        if text[0].islower():
            text = text[0].upper() + text[1:]
        if text[-1] not in '.!?':
            text += '.'
        if identified:
            text = f"{speaker_name}: {text}"

        print(f"✨ {text}")
        paste_text(text)
        print("📋 Pasted!\n")
    except Exception as e:
        print(f"❌ Command transcription error: {e}")


def toggle_command():
    """Toggle command mode recording (double-tap Left Option)."""
    global command_recording, command_buffer

    if not command_recording:
        command_recording = True
        command_buffer = []
        print("🎤 Command mode: recording...")
        message_queue.put({"type": "status", "recording": True, "mode": "command"})
    else:
        command_recording = False
        print("⏹️  Command mode: transcribing...")
        message_queue.put({"type": "status", "recording": False, "mode": "command"})
        threading.Thread(target=command_transcribe_and_paste, daemon=True).start()


# === MEETING MODE (Record button → continuous VAD transcription → file) ===

def toggle_meeting():
    """Toggle meeting mode (Record button in browser)."""
    global meeting_active

    if not meeting_active:
        meeting_active = True
        continuous_transcriber.start_session()
        print("🔴 Meeting mode: recording started")
        message_queue.put({"type": "status", "recording": True, "mode": "meeting"})
    else:
        meeting_active = False
        continuous_transcriber.end_session()
        print("⏹️  Meeting mode: session ended")
        message_queue.put({"type": "status", "recording": False, "mode": "meeting"})


# === KEYBOARD HANDLER ===

def on_press(key):
    """Double-tap Left Option → command mode toggle."""
    global last_option_press

    if key == keyboard.Key.alt_l:
        current_time = time.time()
        if (current_time - last_option_press) < DOUBLE_TAP_THRESHOLD:
            toggle_command()
            last_option_press = 0
        else:
            last_option_press = current_time


def on_release(key):
    pass


# === WEBSOCKET ===

async def websocket_client():
    """WebSocket client that runs in background thread."""
    global ws_connected
    uri = "ws://localhost:3000/ws"

    while True:
        try:
            async with websockets.connect(uri) as websocket:
                ws_connected = True
                print("✅ Connected to server\n")
                # Re-broadcast warmup state so server stays in sync after restarts
                if _warmup_done:
                    message_queue.put({"type": "warmup_complete"})
                else:
                    message_queue.put({"type": "warmup_started"})

                async def send_messages():
                    while True:
                        try:
                            while not message_queue.empty():
                                msg = message_queue.get_nowait()
                                await websocket.send(json.dumps(msg))
                            await asyncio.sleep(0.1)
                        except Exception as e:
                            print(f"Send error: {e}")
                            break

                async def receive_messages():
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            if data.get("type") == "toggle_recording":
                                toggle_meeting()
                            elif data.get("type") == "speaker_db_updated":
                                speaker_name = data.get("speaker_name", "")
                                print(f"🔄 Speaker DB updated ({speaker_name}), reloading…")
                                speaker_db.reload()
                                if continuous_transcriber is not None:
                                    continuous_transcriber.on_speaker_db_updated()
                        except Exception as e:
                            print(f"Receive error: {e}")

                await asyncio.gather(send_messages(), receive_messages(), return_exceptions=True)

        except Exception:
            ws_connected = False
            await asyncio.sleep(5)


def start_websocket_thread():
    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(websocket_client())
    threading.Thread(target=run, daemon=True).start()


# === CONTINUOUS TRANSCRIBER (VAD-based, for meeting mode) ===

class ContinuousTranscriber:
    """VAD-based transcriber for meeting mode. Creates per-session transcript files."""

    SILENCE_THRESHOLD = 0.4
    MAX_SEGMENT_DURATION = 30
    MIN_SPEECH_ENERGY = 100
    TRANSCRIPTS_DIR = Path("transcripts")

    def __init__(self, sample_rate: int, message_queue: queue.Queue,
                 speaker_db: SpeakerDB = None, speaker_identifier: SpeakerIdentifier = None,
                 speaker_diarizer: SpeakerDiarizer = None):
        self._sample_rate = sample_rate
        self._message_queue = message_queue
        self._speaker_db = speaker_db
        self._speaker_identifier = speaker_identifier
        self._speaker_diarizer = speaker_diarizer
        self._session_speaker_map: dict = {}  # pyannote ID → display name
        self._speaker_centroids: dict[str, np.ndarray] = {}  # name -> running centroid (normalized)
        self._speaker_centroid_count: dict[str, int] = {}  # name -> number of embeddings in centroid
        self._buffer: list = []
        self._buffer_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._segment_count: int = 0
        self._session_start_time: datetime | None = None
        self._silence_frames: int = 0
        self._has_speech: bool = False
        self._session_file: Path | None = None
        self._session_dir: Path | None = None
        self._segments_meta: list = []
        # VAD
        import webrtcvad
        self._vad = webrtcvad.Vad(3)
        self._frame_duration_ms = 30
        self._frame_size = int(self._sample_rate * self._frame_duration_ms / 1000)
        self._silence_frames_threshold = int(self.SILENCE_THRESHOLD * 1000 / self._frame_duration_ms)

    def start_session(self) -> None:
        """Start a new meeting session — creates a new transcript file."""
        self._stop_event.clear()
        if self._speaker_db is not None:
            self._speaker_db.reload()
            # Pre-compute centroids from enrolled speakers
            self._speaker_centroids = {}
            self._speaker_centroid_count = {}
            for name, embeddings in self._speaker_db._data.items():
                if embeddings:
                    arr = np.array(embeddings, dtype=np.float32)
                    centroid = arr.mean(axis=0)
                    norm = np.linalg.norm(centroid)
                    if norm > 0:
                        centroid = centroid / norm
                    self._speaker_centroids[name] = centroid
                    self._speaker_centroid_count[name] = len(embeddings)
        self._session_start_time = datetime.now()
        self._segment_count = 0
        self._silence_frames = 0
        self._has_speech = False

        self.TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = self._session_start_time.strftime("%Y-%m-%d_%H-%M-%S")
        self._session_file = self.TRANSCRIPTS_DIR / f"session_{timestamp}.txt"

        with open(self._session_file, "w", encoding="utf-8") as f:
            f.write(f"# Meeting Transcript\n")
            f.write(f"# Started: {self._session_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 40 + "\n")

        self._session_dir = self.TRANSCRIPTS_DIR / self._session_file.stem
        self._session_dir.mkdir(exist_ok=True)
        self._segments_meta = []

        self._session_speaker_map = {}

        self._message_queue.put({
            "type": "session_started",
            "session_file": str(self._session_file)
        })

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def end_session(self) -> None:
        """End the current meeting session — transcribe any remaining audio then stop."""
        self._stop_event.set()

        # Drain and transcribe whatever is still in the buffer (the last utterance
        # before stop is often never cut by VAD because there is no trailing silence).
        with self._buffer_lock:
            remaining = self._buffer
            self._buffer = []

        if remaining:
            try:
                audio_data = np.concatenate(remaining, axis=0)
                rms = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))
                if rms >= self.MIN_SPEECH_ENERGY:
                    elapsed = (datetime.now() - self._session_start_time).total_seconds()
                    audio_float = audio_data.astype(np.float32).flatten() / 32768.0
                    pairs = self._transcribe_pairs(audio_float, audio_data)
                    for text, speaker, turn_audio_float in pairs:
                        self._segment_count += 1
                        display = f"{speaker}: {text}" if speaker else text
                        print(f"📝 [{self._segment_count}] {display}")
                        self._write_segment(display)
                        segment_id = f"seg_{self._segment_count:03d}"
                        self._message_queue.put({
                            "type": "transcript_segment",
                            "text": text,
                            "speaker": speaker,
                            "segment_id": segment_id,
                            "timestamp": datetime.now().isoformat()
                        })
                        turn_int16 = (turn_audio_float * 32768.0).astype(np.int16)
                        audio_path = self._save_segment_audio(turn_int16, segment_id)
                        self._segments_meta.append({
                            "id": segment_id,
                            "timestamp": datetime.now().isoformat(),
                            "elapsed": self._format_duration(elapsed),
                            "text": text,
                            "speaker": speaker,
                            "audio_path": audio_path,
                        })
            except Exception as e:
                print(f"⚠️  Final segment transcription failed: {e}")

        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

        # Write footer
        if self._session_file:
            duration = (datetime.now() - self._session_start_time).total_seconds()
            with open(self._session_file, "a", encoding="utf-8") as f:
                f.write("-" * 40 + "\n")
                f.write(f"# Duration: {self._format_duration(duration)}\n")
                f.write(f"# Segments: {self._segment_count}\n")

            json_path = self._session_file.with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "session_id": self._session_file.stem,
                    "started": self._session_start_time.isoformat(),
                    "ended": datetime.now().isoformat(),
                    "segment_count": self._segment_count,
                    "segments": self._segments_meta,
                }, f, indent=2, ensure_ascii=False)

            self._message_queue.put({
                "type": "session_ended",
                "duration_seconds": duration,
                "segment_count": self._segment_count,
                "session_file": str(self._session_file)
            })

    def feed(self, audio_chunk: np.ndarray) -> None:
        """Feed audio from callback."""
        with self._buffer_lock:
            self._buffer.append(audio_chunk)

    def _run_loop(self) -> None:
        """VAD loop: cut on silence, transcribe, write."""
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=0.1)

            with self._buffer_lock:
                if not self._buffer:
                    continue
                current_buffer = list(self._buffer)

            total_audio = np.concatenate(current_buffer, axis=0).flatten()
            if len(total_audio) < self._frame_size:
                continue

            # Check last frame for speech/silence
            last_frame = total_audio[-self._frame_size:]
            frame_bytes = last_frame.astype(np.int16).tobytes()
            try:
                is_speech = self._vad.is_speech(frame_bytes, self._sample_rate)
            except Exception:
                is_speech = True

            if is_speech:
                self._has_speech = True
                self._silence_frames = 0
            else:
                self._silence_frames += 1

            total_duration = len(total_audio) / self._sample_rate
            should_cut = (
                (self._has_speech and self._silence_frames >= self._silence_frames_threshold)
                or total_duration >= self.MAX_SEGMENT_DURATION
            )

            if should_cut and total_duration >= 0.5:
                with self._buffer_lock:
                    captured = self._buffer
                    self._buffer = []
                audio_data = np.concatenate(captured, axis=0)
                self._silence_frames = 0
                self._has_speech = False

                rms = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))
                if rms < self.MIN_SPEECH_ENERGY:
                    continue

                elapsed_at_cut = (datetime.now() - self._session_start_time).total_seconds()
                audio_float = audio_data.astype(np.float32).flatten() / 32768.0
                pairs = self._transcribe_pairs(audio_float, audio_data)

                for text, speaker, turn_audio_float in pairs:
                    self._segment_count += 1
                    display = f"{speaker}: {text}" if speaker else text
                    print(f"📝 [{self._segment_count}] {display}")
                    self._write_segment(display)
                    segment_id = f"seg_{self._segment_count:03d}"
                    self._message_queue.put({
                        "type": "transcript_segment",
                        "text": text,
                        "speaker": speaker,
                        "segment_id": segment_id,
                        "timestamp": datetime.now().isoformat()
                    })
                    # Save the speaker turn audio (or full segment if no diarization)
                    turn_int16 = (turn_audio_float * 32768.0).astype(np.int16)
                    audio_path = self._save_segment_audio(turn_int16, segment_id)
                    self._segments_meta.append({
                        "id": segment_id,
                        "timestamp": datetime.now().isoformat(),
                        "elapsed": self._format_duration(elapsed_at_cut),
                        "text": text,
                        "speaker": speaker,
                        "audio_path": audio_path,
                    })

    def _save_segment_audio(self, audio_data: np.ndarray, segment_id: str) -> str:
        """Write a segment's audio to a WAV file in the session directory. Returns relative path."""
        path = self._session_dir / f"{segment_id}.wav"
        with wave.open(str(path), 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16 = 2 bytes
            wf.setframerate(self._sample_rate)
            wf.writeframes(audio_data.astype(np.int16).flatten().tobytes())
        return str(Path(self._session_dir.name) / f"{segment_id}.wav")

    def _transcribe_pairs(
        self, audio_float: np.ndarray, audio_int16: np.ndarray
    ) -> list:
        """Transcribe audio, splitting by speaker turns when diarization is available.

        Returns list of (text, speaker_name_or_None, turn_audio_float32).
        """
        try:
            if self._speaker_diarizer is not None:
                turns = self._speaker_diarizer.diarize(audio_float, self._sample_rate)
                if len(turns) > 1:
                    results = []
                    for start, end, spk_id in turns:
                        s = int(start * self._sample_rate)
                        e = int(end * self._sample_rate)
                        turn_audio = audio_float[s:e]
                        if len(turn_audio) < int(0.3 * self._sample_rate):
                            continue
                        text = transcribe_audio(turn_audio)
                        if text:
                            speaker = self._resolve_speaker(spk_id, turn_audio)
                            results.append((text, speaker, turn_audio))
                    if results:
                        return results

            # Single-speaker fallback (no diarizer, or only one speaker detected)
            text = transcribe_audio(audio_float)
            if not text:
                return []
            speaker = None
            if (self._speaker_identifier is not None and self._speaker_centroids):
                embedding = self._speaker_identifier.get_embedding(audio_float, self._sample_rate)
                if not np.all(embedding == 0):
                    emb_flat = embedding.flatten()
                    best_name = None
                    best_sim = -1.0
                    for name, centroid in self._speaker_centroids.items():
                        sim = 1 - cdist(emb_flat.reshape(1, -1), centroid.reshape(1, -1), metric='cosine')[0, 0]
                        if sim > best_sim:
                            best_sim = sim
                            best_name = name
                    if best_name and best_sim >= self._speaker_identifier.similarity_threshold:
                        speaker = best_name
                        # Update rolling centroid with exponential moving average
                        count = self._speaker_centroid_count.get(best_name, 1)
                        alpha = 1.0 / (count + 1)
                        new_centroid = (1 - alpha) * self._speaker_centroids[best_name] + alpha * emb_flat
                        norm = np.linalg.norm(new_centroid)
                        if norm > 0:
                            new_centroid = new_centroid / norm
                        self._speaker_centroids[best_name] = new_centroid
                        self._speaker_centroid_count[best_name] = count + 1
            return [(text, speaker, audio_float)]
        except Exception as e:
            print(f"⚠️  Transcription error: {e}")
            return []

    def on_speaker_db_updated(self) -> None:
        """Called when the speaker DB is updated externally (e.g. via annotation).
        Reload DB and clear anonymous speaker mappings so they get re-identified."""
        if self._speaker_db is not None:
            self._speaker_db.reload()
            # Rebuild centroids from updated DB
            for name, embeddings in self._speaker_db._data.items():
                if embeddings:
                    arr = np.array(embeddings, dtype=np.float32)
                    centroid = arr.mean(axis=0)
                    norm = np.linalg.norm(centroid)
                    if norm > 0:
                        centroid = centroid / norm
                    self._speaker_centroids[name] = centroid
                    self._speaker_centroid_count[name] = len(embeddings)
        # Clear anonymous entries so the next segment from that speaker gets re-identified
        anonymous_keys = [
            k for k, v in self._session_speaker_map.items()
            if v is None or re.match(r'^Speaker \d+$', v)
        ]
        for k in anonymous_keys:
            del self._session_speaker_map[k]

    def _resolve_speaker(self, spk_id: str, turn_audio: np.ndarray) -> str:
        """Map a pyannote speaker ID to a display name, consistent within the session."""
        if spk_id in self._session_speaker_map:
            return self._session_speaker_map[spk_id]

        # Try to match against enrolled speakers using rolling centroids
        if (self._speaker_identifier is not None and self._speaker_db is not None
                and self._speaker_centroids):
            embedding = self._speaker_identifier.get_embedding(turn_audio, self._sample_rate)
            if not np.all(embedding == 0):
                emb_flat = embedding.flatten()
                best_name = None
                best_sim = -1.0
                for name, centroid in self._speaker_centroids.items():
                    sim = 1 - cdist(emb_flat.reshape(1, -1), centroid.reshape(1, -1), metric='cosine')[0, 0]
                    if sim > best_sim:
                        best_sim = sim
                        best_name = name

                if best_name and best_sim >= self._speaker_identifier.similarity_threshold:
                    # Update rolling centroid with exponential moving average
                    count = self._speaker_centroid_count.get(best_name, 1)
                    alpha = 1.0 / (count + 1)
                    new_centroid = (1 - alpha) * self._speaker_centroids[best_name] + alpha * emb_flat
                    norm = np.linalg.norm(new_centroid)
                    if norm > 0:
                        new_centroid = new_centroid / norm
                    self._speaker_centroids[best_name] = new_centroid
                    self._speaker_centroid_count[best_name] = count + 1

                    self._session_speaker_map[spk_id] = best_name
                    return best_name

        # Assign anonymous sequential label
        n = len(self._session_speaker_map) + 1
        label = f"Speaker {n}"
        self._session_speaker_map[spk_id] = label
        return label

    def _write_segment(self, text: str) -> None:
        """Append segment to session file."""
        try:
            elapsed = (datetime.now() - self._session_start_time).total_seconds()
            ts = self._format_duration(elapsed)
            with open(self._session_file, "a", encoding="utf-8") as f:
                f.write(f"[{ts}] {text}\n")
        except Exception as e:
            print(f"⚠️  Write failed: {e}")

    @staticmethod
    def _format_duration(seconds: float) -> str:
        s = int(seconds)
        return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


# === MAIN ===

def main():
    global continuous_transcriber

    print("🎙️  Voice Inject")
    print("   Meeting mode: Click Record in browser (VAD-based, saves transcript)")
    print("   Command mode: Double-tap Left Option ⌥ (transcribe → paste)")
    print("   Press Ctrl+C to quit.\n")

    start_websocket_thread()

    def sigint_handler(signum, frame):
        print("\n⏹️  Shutting down...")
        if meeting_active and continuous_transcriber:
            continuous_transcriber.end_session()
        sys.exit(0)
    signal.signal(signal.SIGINT, sigint_handler)

    continuous_transcriber = ContinuousTranscriber(
        sample_rate=SAMPLE_RATE,
        message_queue=message_queue,
        speaker_db=speaker_db,
        speaker_identifier=speaker_identifier,
        speaker_diarizer=speaker_diarizer,
    )

    threading.Thread(target=_warmup_models, daemon=True).start()

    print("🎙️  Ready\n")

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype="int16", callback=audio_callback):
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()


if __name__ == "__main__":
    main()

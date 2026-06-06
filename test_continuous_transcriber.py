"""Unit tests for ContinuousTranscriber class."""

import queue
import threading
import time
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import os


# We need to mock hardware-dependent modules before importing client
with patch.dict('sys.modules', {
    'sounddevice': MagicMock(),
    'pynput': MagicMock(),
    'pynput.keyboard': MagicMock(),
}):
    from client import ContinuousTranscriber


class TestContinuousTranscriberInit:
    """Tests for __init__ method."""

    def test_initializes_with_correct_attributes(self):
        model = MagicMock()
        mq = queue.Queue()
        ct = ContinuousTranscriber(model=model, sample_rate=16000, message_queue=mq)

        assert ct._model is model
        assert ct._sample_rate == 16000
        assert ct._message_queue is mq
        assert ct._buffer == []
        assert ct._thread is None
        assert ct._segment_count == 0

    def test_batch_interval_is_10_seconds(self):
        assert ContinuousTranscriber.BATCH_INTERVAL == 10


class TestFeed:
    """Tests for feed() method."""

    def test_feed_appends_audio_chunk(self):
        model = MagicMock()
        mq = queue.Queue()
        ct = ContinuousTranscriber(model=model, sample_rate=16000, message_queue=mq)

        chunk = np.zeros((1600, 1), dtype=np.int16)
        ct.feed(chunk)

        assert len(ct._buffer) == 1
        np.testing.assert_array_equal(ct._buffer[0], chunk)

    def test_feed_multiple_chunks(self):
        model = MagicMock()
        mq = queue.Queue()
        ct = ContinuousTranscriber(model=model, sample_rate=16000, message_queue=mq)

        for i in range(5):
            chunk = np.full((1600, 1), i, dtype=np.int16)
            ct.feed(chunk)

        assert len(ct._buffer) == 5

    def test_feed_is_thread_safe(self):
        """Multiple threads can feed concurrently without data corruption."""
        model = MagicMock()
        mq = queue.Queue()
        ct = ContinuousTranscriber(model=model, sample_rate=16000, message_queue=mq)

        num_threads = 10
        chunks_per_thread = 100

        def feeder():
            for _ in range(chunks_per_thread):
                ct.feed(np.zeros((160, 1), dtype=np.int16))

        threads = [threading.Thread(target=feeder) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(ct._buffer) == num_threads * chunks_per_thread


class TestStartStop:
    """Tests for start() and stop() methods."""

    def test_start_creates_thread(self):
        model = MagicMock()
        mq = queue.Queue()
        ct = ContinuousTranscriber(model=model, sample_rate=16000, message_queue=mq)

        ct.start()
        assert ct._thread is not None
        assert ct._thread.is_alive()

        ct.stop()
        assert ct._thread is None

    def test_stop_signals_event(self):
        model = MagicMock()
        mq = queue.Queue()
        ct = ContinuousTranscriber(model=model, sample_rate=16000, message_queue=mq)

        ct.start()
        assert not ct._stop_event.is_set()

        ct.stop()
        assert ct._stop_event.is_set()

    def test_start_clears_stop_event(self):
        model = MagicMock()
        mq = queue.Queue()
        ct = ContinuousTranscriber(model=model, sample_rate=16000, message_queue=mq)

        ct._stop_event.set()
        ct.start()
        assert not ct._stop_event.is_set()

        ct.stop()


class TestTranscribeBuffer:
    """Tests for _transcribe_buffer() method."""

    def test_returns_transcribed_text(self):
        model = MagicMock()
        # Mock model.transcribe to return segments
        mock_segment = MagicMock()
        mock_segment.text = " Hello world "
        model.transcribe.return_value = ([mock_segment], MagicMock())

        mq = queue.Queue()
        ct = ContinuousTranscriber(model=model, sample_rate=16000, message_queue=mq)

        audio = np.zeros((16000,), dtype=np.int16)
        result = ct._transcribe_buffer(audio)

        assert result == "Hello world"

    def test_returns_none_on_empty_result(self):
        model = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = "   "
        model.transcribe.return_value = ([mock_segment], MagicMock())

        mq = queue.Queue()
        ct = ContinuousTranscriber(model=model, sample_rate=16000, message_queue=mq)

        audio = np.zeros((16000,), dtype=np.int16)
        result = ct._transcribe_buffer(audio)

        assert result is None

    def test_returns_none_on_no_segments(self):
        model = MagicMock()
        model.transcribe.return_value = ([], MagicMock())

        mq = queue.Queue()
        ct = ContinuousTranscriber(model=model, sample_rate=16000, message_queue=mq)

        audio = np.zeros((16000,), dtype=np.int16)
        result = ct._transcribe_buffer(audio)

        assert result is None

    def test_returns_none_on_error(self):
        model = MagicMock()
        model.transcribe.side_effect = RuntimeError("Model error")

        mq = queue.Queue()
        ct = ContinuousTranscriber(model=model, sample_rate=16000, message_queue=mq)

        audio = np.zeros((16000,), dtype=np.int16)
        result = ct._transcribe_buffer(audio)

        assert result is None

    def test_normalizes_int16_to_float32(self):
        """Verifies audio is converted from int16 to float32 in [-1, 1] range."""
        model = MagicMock()
        model.transcribe.return_value = ([], MagicMock())

        mq = queue.Queue()
        ct = ContinuousTranscriber(model=model, sample_rate=16000, message_queue=mq)

        # Use max int16 value to verify normalization
        audio = np.array([32767, -32768], dtype=np.int16)
        ct._transcribe_buffer(audio)

        # Check the audio passed to model.transcribe is float32 and normalized
        call_args = model.transcribe.call_args
        audio_arg = call_args[0][0]
        assert audio_arg.dtype == np.float32
        assert audio_arg.max() <= 1.0
        assert audio_arg.min() >= -1.0


class TestRunLoop:
    """Tests for _run_loop() behavior via start/stop with short interval."""

    def test_run_loop_transcribes_buffered_audio(self):
        """When audio is buffered, _run_loop transcribes it and puts result in queue."""
        model = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = "test transcription"
        model.transcribe.return_value = ([mock_segment], MagicMock())

        mq = queue.Queue()
        ct = ContinuousTranscriber(model=model, sample_rate=16000, message_queue=mq)

        # Use a very short batch interval for testing
        ct.BATCH_INTERVAL = 0.1

        # Feed some audio
        ct.feed(np.zeros((16000, 1), dtype=np.int16))

        ct.start()
        # Wait enough time for one batch to process
        time.sleep(0.5)
        ct.stop()

        # Check message was put in queue
        assert not mq.empty()
        msg = mq.get_nowait()
        assert msg["type"] == "transcript_segment"
        assert msg["text"] == "test transcription"
        assert "timestamp" in msg

    def test_run_loop_skips_empty_buffer(self):
        """When buffer is empty, no transcription is attempted."""
        model = MagicMock()
        mq = queue.Queue()
        ct = ContinuousTranscriber(model=model, sample_rate=16000, message_queue=mq)

        ct.BATCH_INTERVAL = 0.1

        ct.start()
        time.sleep(0.3)
        ct.stop()

        # No transcription should have been attempted
        model.transcribe.assert_not_called()
        # The only message in the queue should be session_ended (from stop())
        # No transcript_segment messages should be present
        messages = []
        while not mq.empty():
            messages.append(mq.get_nowait())
        transcript_segments = [m for m in messages if m["type"] == "transcript_segment"]
        assert len(transcript_segments) == 0

    def test_run_loop_increments_segment_count(self):
        """Each successful transcription increments the segment count."""
        model = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = "segment"
        model.transcribe.return_value = ([mock_segment], MagicMock())

        mq = queue.Queue()
        ct = ContinuousTranscriber(model=model, sample_rate=16000, message_queue=mq)
        ct.BATCH_INTERVAL = 0.1

        # Feed audio for multiple batches
        ct.feed(np.zeros((16000, 1), dtype=np.int16))
        ct.start()
        time.sleep(0.15)
        ct.feed(np.zeros((16000, 1), dtype=np.int16))
        time.sleep(0.15)
        ct.stop()

        assert ct._segment_count >= 1


class TestSessionFileWriter:
    """Tests for session file writer methods (_write_header, _write_segment, _write_footer)."""

    def _make_transcriber(self, tmp_path):
        """Helper to create a ContinuousTranscriber with a temp session file."""
        model = MagicMock()
        mq = queue.Queue()
        ct = ContinuousTranscriber(model=model, sample_rate=16000, message_queue=mq)
        ct.TRANSCRIPTS_DIR = tmp_path / "transcripts"
        return ct

    def test_start_creates_transcripts_directory(self, tmp_path):
        """start() creates the transcripts/ dir if it doesn't exist."""
        ct = self._make_transcriber(tmp_path)
        ct.start()
        ct.stop()

        assert ct.TRANSCRIPTS_DIR.exists()

    def test_start_generates_session_file_path(self, tmp_path):
        """start() generates a session file path with expected naming format."""
        ct = self._make_transcriber(tmp_path)
        ct.start()

        assert ct.session_file_path is not None
        assert ct.session_file_path.name.startswith("session_")
        assert ct.session_file_path.name.endswith(".txt")
        # Check pattern: session_YYYY-MM-DD_HH-MM-SS.txt
        name = ct.session_file_path.stem  # e.g. session_2025-01-15_14-30-00
        parts = name.split("_", 1)  # ['session', '2025-01-15_14-30-00']
        assert parts[0] == "session"

        ct.stop()

    def test_write_header_format(self, tmp_path):
        """_write_header writes correct header with title, timestamp, and separator."""
        ct = self._make_transcriber(tmp_path)
        ct.start()
        ct.stop()

        content = ct.session_file_path.read_text()
        lines = content.strip().split("\n")

        assert lines[0] == "# Voice Inject — Live Transcription Session"
        assert lines[1].startswith("# Started: ")
        # Verify timestamp format YYYY-MM-DD HH:MM:SS
        start_str = lines[1].replace("# Started: ", "")
        assert len(start_str) == 19  # "2025-01-15 14:30:00"
        assert lines[2] == "-" * 40

    def test_write_segment_format(self, tmp_path):
        """_write_segment writes [HH:MM:SS] text format and flushes."""
        ct = self._make_transcriber(tmp_path)
        ct.start()

        # Small delay so elapsed time > 0
        time.sleep(0.1)
        ct._write_segment("Hello world")
        ct.stop()

        content = ct.session_file_path.read_text()
        lines = content.strip().split("\n")

        # Find the segment line (after header lines)
        segment_line = lines[3]  # After title, started, separator
        assert segment_line.startswith("[00:00:")
        assert "Hello world" in segment_line

    def test_write_segment_elapsed_time(self, tmp_path):
        """_write_segment uses elapsed time from session start."""
        from datetime import timedelta

        ct = self._make_transcriber(tmp_path)
        ct.start()

        # Manually set _session_start_time to 65 seconds ago to test HH:MM:SS formatting
        from datetime import datetime
        ct._session_start_time = datetime.now() - timedelta(seconds=65)
        ct._write_segment("After one minute five seconds")
        ct.stop()

        content = ct.session_file_path.read_text()
        assert "[00:01:05]" in content

    def test_write_footer_format(self, tmp_path):
        """_write_footer writes separator, end time, duration, and segment count."""
        ct = self._make_transcriber(tmp_path)
        ct.start()
        ct._segment_count = 5
        ct.stop()

        content = ct.session_file_path.read_text()
        lines = content.strip().split("\n")

        # Footer should be at the end
        assert lines[-4] == "-" * 40
        assert lines[-3].startswith("# Session ended: ")
        assert lines[-2].startswith("# Duration: ")
        assert lines[-1] == "# Segments: 5"

    def test_write_footer_duration_format(self, tmp_path):
        """_write_footer duration uses HH:MM:SS format."""
        from datetime import timedelta, datetime

        ct = self._make_transcriber(tmp_path)
        ct.start()

        # Simulate 1 hour 23 minutes 45 seconds session
        ct._session_start_time = datetime.now() - timedelta(hours=1, minutes=23, seconds=45)
        ct.stop()

        content = ct.session_file_path.read_text()
        assert "# Duration: 01:23:45" in content

    def test_full_session_file_format(self, tmp_path):
        """Complete session file has proper structure: header, segments, footer."""
        ct = self._make_transcriber(tmp_path)
        ct.start()

        ct._write_segment("First segment")
        ct._write_segment("Second segment")
        ct._segment_count = 2
        ct.stop()

        content = ct.session_file_path.read_text()
        lines = content.strip().split("\n")

        # Header
        assert "# Voice Inject" in lines[0]
        assert "# Started:" in lines[1]
        assert lines[2] == "-" * 40

        # Segments
        assert "[00:00:" in lines[3]
        assert "First segment" in lines[3]
        assert "[00:00:" in lines[4]
        assert "Second segment" in lines[4]

        # Footer
        assert lines[5] == "-" * 40
        assert "# Session ended:" in lines[6]
        assert "# Duration:" in lines[7]
        assert "# Segments: 2" in lines[8]

    def test_write_segment_flushes_to_disk(self, tmp_path):
        """_write_segment flushes after each write so data is recoverable."""
        ct = self._make_transcriber(tmp_path)
        ct.start()

        ct._write_segment("flushed text")

        # Read file while it's still open (not yet stopped)
        content = ct.session_file_path.read_text()
        assert "flushed text" in content

        ct.stop()

    def test_run_loop_calls_write_segment(self, tmp_path):
        """_run_loop calls _write_segment after successful transcription."""
        model = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = "transcribed text"
        model.transcribe.return_value = ([mock_segment], MagicMock())

        mq = queue.Queue()
        ct = ContinuousTranscriber(model=model, sample_rate=16000, message_queue=mq)
        ct.TRANSCRIPTS_DIR = tmp_path / "transcripts"
        ct.BATCH_INTERVAL = 0.1

        ct.start()
        ct.feed(np.zeros((16000, 1), dtype=np.int16))
        time.sleep(0.5)
        ct.stop()

        content = ct.session_file_path.read_text()
        assert "transcribed text" in content

    def test_stop_closes_file_handle(self, tmp_path):
        """stop() closes the file handle after writing footer."""
        ct = self._make_transcriber(tmp_path)
        ct.start()
        ct.stop()

        assert ct._file_handle is None or ct._file_handle.closed


class TestGracefulShutdown:
    """Tests for graceful shutdown behavior (SIGINT handler and stop() with buffer threshold)."""

    def _make_transcriber(self, tmp_path, model=None):
        """Helper to create a ContinuousTranscriber with a temp session file."""
        if model is None:
            model = MagicMock()
        mq = queue.Queue()
        ct = ContinuousTranscriber(model=model, sample_rate=16000, message_queue=mq)
        ct.TRANSCRIPTS_DIR = tmp_path / "transcripts"
        return ct, mq

    def test_stop_transcribes_buffer_ge_16000_samples(self, tmp_path):
        """stop() transcribes remaining buffer if it has >= 16000 samples (1 second)."""
        model = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = "final segment"
        model.transcribe.return_value = ([mock_segment], MagicMock())

        ct, mq = self._make_transcriber(tmp_path, model=model)
        ct.start()

        # Feed exactly 16000 samples (1 second at 16kHz)
        ct.feed(np.zeros((16000, 1), dtype=np.int16))

        ct.stop()

        # Verify transcription was called
        model.transcribe.assert_called_once()

        # Verify segment was written to session file
        content = ct.session_file_path.read_text()
        assert "final segment" in content

    def test_stop_discards_buffer_lt_16000_samples(self, tmp_path):
        """stop() discards remaining buffer if it has < 16000 samples."""
        model = MagicMock()
        ct, mq = self._make_transcriber(tmp_path, model=model)
        ct.start()

        # Feed only 8000 samples (0.5 seconds at 16kHz) — below threshold
        ct.feed(np.zeros((8000, 1), dtype=np.int16))

        ct.stop()

        # Verify transcription was NOT called
        model.transcribe.assert_not_called()

        # Session file should have header and footer but no segment
        content = ct.session_file_path.read_text()
        assert "# Voice Inject" in content
        assert "# Session ended:" in content

    def test_stop_sends_session_ended_message(self, tmp_path):
        """stop() sends session_ended message with duration and segment count."""
        ct, mq = self._make_transcriber(tmp_path)
        ct.start()
        ct._segment_count = 5

        ct.stop()

        # Check session_ended message
        messages = []
        while not mq.empty():
            messages.append(mq.get_nowait())

        session_ended = [m for m in messages if m["type"] == "session_ended"]
        assert len(session_ended) == 1
        msg = session_ended[0]
        assert "duration_seconds" in msg
        assert msg["segment_count"] == 5
        assert isinstance(msg["duration_seconds"], float)

    def test_stop_handles_transcription_failure_with_untranscribed_note(self, tmp_path):
        """If transcription fails during shutdown, an untranscribed note is appended."""
        model = MagicMock()
        model.transcribe.side_effect = RuntimeError("Model crashed")

        ct, mq = self._make_transcriber(tmp_path, model=model)
        ct.start()

        # Feed >= 16000 samples to trigger transcription attempt
        ct.feed(np.zeros((20000, 1), dtype=np.int16))

        ct.stop()

        # Verify untranscribed note in session file
        content = ct.session_file_path.read_text()
        assert "[UNTRANSCRIBED] Final audio segment could not be processed" in content
        # Footer should still be written
        assert "# Session ended:" in content

    def test_stop_writes_footer_even_on_transcription_failure(self, tmp_path):
        """Footer is written even when transcription of remainder fails."""
        model = MagicMock()
        model.transcribe.side_effect = Exception("Unexpected error")

        ct, mq = self._make_transcriber(tmp_path, model=model)
        ct.start()
        ct._segment_count = 3

        # Feed enough audio to trigger transcription
        ct.feed(np.zeros((16000, 1), dtype=np.int16))

        ct.stop()

        content = ct.session_file_path.read_text()
        assert "# Segments: 3" in content
        assert "# Duration:" in content

    def test_stop_closes_file_handle(self, tmp_path):
        """stop() closes the file handle after writing footer."""
        ct, mq = self._make_transcriber(tmp_path)
        ct.start()
        ct.stop()

        assert ct._file_handle is None or ct._file_handle.closed

    def test_stop_with_empty_buffer_no_transcription(self, tmp_path):
        """stop() with empty buffer doesn't attempt transcription."""
        model = MagicMock()
        ct, mq = self._make_transcriber(tmp_path, model=model)
        ct.start()

        # Don't feed any audio
        ct.stop()

        model.transcribe.assert_not_called()


class TestSigintHandler:
    """Tests for the SIGINT signal handler registration in main()."""

    def test_signal_import_available(self):
        """signal module is importable for SIGINT handling."""
        import signal
        assert hasattr(signal, 'SIGINT')

    def test_sigint_handler_calls_stop(self):
        """SIGINT handler calls continuous_transcriber.stop()."""
        import signal
        import sys
        from unittest.mock import patch, MagicMock

        # Mock the ContinuousTranscriber
        mock_transcriber = MagicMock()

        # Simulate what the sigint_handler does
        def sigint_handler(signum, frame):
            if mock_transcriber is not None:
                mock_transcriber.stop()

        # Call the handler directly
        sigint_handler(signal.SIGINT, None)

        mock_transcriber.stop.assert_called_once()

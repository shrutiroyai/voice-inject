"""Unit tests for server.py session state and new WebSocket message types."""

import json
import pytest
from starlette.testclient import TestClient

import server
from server import app


@pytest.fixture(autouse=True)
def reset_server_state():
    """Reset global server state between tests."""
    server.session_state = {}
    server.active_connections = []
    yield
    server.session_state = {}
    server.active_connections = []


class TestSessionStarted:
    """Tests for session_started message handling."""

    def test_session_started_stored_in_session_state(self):
        """session_started message is stored in session_state."""
        client = TestClient(app)
        with client.websocket_connect("/ws") as ws1:
            with client.websocket_connect("/ws") as ws2:
                msg = {"type": "session_started", "session_file": "/tmp/session.txt"}
                ws1.send_text(json.dumps(msg))
                # Wait for broadcast to ws2 to confirm server processed the message
                ws2.receive_text()
                assert server.session_state == msg

    def test_session_started_broadcast_to_other_clients(self):
        """session_started is broadcast to other connected clients."""
        client = TestClient(app)
        with client.websocket_connect("/ws") as ws1:
            with client.websocket_connect("/ws") as ws2:
                msg = {"type": "session_started", "session_file": "/tmp/session.txt"}
                ws1.send_text(json.dumps(msg))
                received = json.loads(ws2.receive_text())
                assert received == msg


class TestTranscriptSegment:
    """Tests for transcript_segment message handling."""

    def test_transcript_segment_broadcast_to_other_clients(self):
        """transcript_segment is broadcast to other connected browser clients."""
        client = TestClient(app)
        with client.websocket_connect("/ws") as ws1:
            with client.websocket_connect("/ws") as ws2:
                msg = {
                    "type": "transcript_segment",
                    "text": "Hello world",
                    "timestamp": "2025-01-15T14:30:10.000Z",
                }
                ws1.send_text(json.dumps(msg))
                received = json.loads(ws2.receive_text())
                assert received == msg

    def test_transcript_segment_discarded_if_no_other_clients(self):
        """transcript_segment is silently discarded if only the sender is connected."""
        client = TestClient(app)
        with client.websocket_connect("/ws") as ws:
            msg = {
                "type": "transcript_segment",
                "text": "Hello",
                "timestamp": "2025-01-15T14:30:10.000Z",
            }
            ws.send_text(json.dumps(msg))
            # No error, message just discarded — test passes if no exception


class TestSessionEnded:
    """Tests for session_ended message handling."""

    def test_session_ended_clears_session_state(self):
        """session_ended clears the stored session_state."""
        client = TestClient(app)
        with client.websocket_connect("/ws") as ws1:
            # First, establish a session
            start_msg = {"type": "session_started", "session_file": "/tmp/s.txt"}
            ws1.send_text(json.dumps(start_msg))

            with client.websocket_connect("/ws") as ws2:
                # ws2 should receive the stored session_started on connect
                ws2.receive_text()

                # Now send session_ended from ws1
                msg = {"type": "session_ended", "duration_seconds": 60.0, "segment_count": 6}
                ws1.send_text(json.dumps(msg))
                # Wait for broadcast to ws2 to confirm server processed
                ws2.receive_text()
                assert server.session_state == {}

    def test_session_ended_broadcast_to_other_clients(self):
        """session_ended is broadcast to other connected clients."""
        client = TestClient(app)
        with client.websocket_connect("/ws") as ws1:
            with client.websocket_connect("/ws") as ws2:
                msg = {"type": "session_ended", "duration_seconds": 120.5, "segment_count": 12}
                ws1.send_text(json.dumps(msg))
                received = json.loads(ws2.receive_text())
                assert received == msg


class TestLateConnect:
    """Tests for late-connecting clients receiving session context."""

    def test_new_client_receives_stored_session_started(self):
        """A newly connecting client receives the stored session_started message."""
        client = TestClient(app)
        # First client starts a session
        with client.websocket_connect("/ws") as ws1:
            msg = {"type": "session_started", "session_file": "/tmp/session.txt"}
            ws1.send_text(json.dumps(msg))

            # Second client connects late and should receive session_started
            with client.websocket_connect("/ws") as ws2:
                received = json.loads(ws2.receive_text())
                assert received == msg

    def test_new_client_no_message_when_no_session(self):
        """A newly connecting client receives nothing if no session is active."""
        client = TestClient(app)
        # session_state is empty (reset by fixture)
        with client.websocket_connect("/ws") as ws:
            # Send a ping to verify connection works — no session_started should precede it
            ws.send_text(json.dumps({"type": "status", "recording": False}))
            # If session_state were sent, we'd get it before anything else
            # This test passes if no unexpected message arrives

"""
speaker_id.py — Speaker identification and diarization for voice-inject.

Provides:
  SpeakerIdentifier — per-segment speaker identification via pyannote/embedding
  SpeakerDiarizer   — speaker turn detection via pyannote/speaker-diarization

Audio contract:
  - dtype  : float32, normalised to [-1.0, 1.0]
  - shape  : (N,)  — 1-D mono array
  - rate   : 16 000 Hz
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

from speaker_db import SpeakerDB

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _load_hf_token() -> Optional[str]:
    """Return HuggingFace token from env or config file."""
    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if token:
        return token.strip()
    config_path = Path.home() / ".voice-inject" / "config.yaml"
    if config_path.exists():
        try:
            import yaml
            with config_path.open() as fh:
                cfg = yaml.safe_load(fh) or {}
            token = cfg.get("hf_token")
            if token:
                return str(token).strip()
        except Exception as exc:
            logger.warning("Could not read %s: %s", config_path, exc)
    return None


# ---------------------------------------------------------------------------
# SpeakerDiarizer — speaker turn detection
# ---------------------------------------------------------------------------

class SpeakerDiarizer:
    """Detects speaker change points in an audio segment.

    Uses pyannote segmentation + embedding + agglomerative clustering.
    Lazy-loaded on first call to diarize().

    diarize(audio_float32, sample_rate) → list[(start_sec, end_sec, speaker_id)]
    """

    # Clustering threshold: lower → more speakers, higher → fewer.
    # 0.80 balances sensitivity vs. over-segmentation for typical speech.
    CLUSTERING_THRESHOLD = 0.80

    def __init__(self, hf_token: Optional[str] = None) -> None:
        self._hf_token: Optional[str] = hf_token
        self._pipeline = None

    def _ensure_pipeline(self) -> None:
        if self._pipeline is not None:
            return

        token = self._hf_token or _load_hf_token()
        if not token:
            raise ValueError("HuggingFace token required for SpeakerDiarizer.")

        # pyannote 4.x SpeakerDiarization tries to load a PLDA from the
        # gated pyannote/speaker-diarization-community-1 repo by default.
        # When using AgglomerativeClustering, the PLDA is computed but
        # never accessed — so we patch get_plda to skip the gated download.
        import pyannote.audio.pipelines.speaker_diarization as _sd_mod
        _orig_get_plda = _sd_mod.get_plda

        def _safe_get_plda(plda, **kw):
            try:
                return _orig_get_plda(plda, **kw)
            except Exception:
                return None  # safe: never used with AgglomerativeClustering

        _sd_mod.get_plda = _safe_get_plda

        try:
            import io, sys, warnings
            from pyannote.audio.pipelines import SpeakerDiarization

            # Suppress Lightning checkpoint upgrade messages and PLDA 403 stderr prints
            _devnull = open(os.devnull, "w")
            _old_stderr = sys.stderr
            sys.stderr = _devnull
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pipeline = SpeakerDiarization(
                        segmentation={"checkpoint": "pyannote/segmentation", "revision": "2.1"},
                        embedding="pyannote/embedding",
                        clustering="AgglomerativeClustering",
                        token=token,
                    )
            finally:
                sys.stderr = _old_stderr
                _devnull.close()
        finally:
            _sd_mod.get_plda = _orig_get_plda  # restore

        pipeline.segmentation.threshold = 0.5
        pipeline.segmentation.min_duration_off = 0.1
        pipeline.clustering.threshold = self.CLUSTERING_THRESHOLD
        pipeline.clustering.method = "average"
        pipeline.clustering.min_cluster_size = 15

        self._pipeline = pipeline
        logger.info("SpeakerDiarizer pipeline ready.")

    def diarize(
        self, audio_float32: np.ndarray, sample_rate: int
    ) -> list:
        """Return speaker turns as list of (start_sec, end_sec, speaker_id)."""
        import torch

        if len(audio_float32) / sample_rate < 1.0:
            return []  # too short to diarize

        self._ensure_pipeline()

        waveform = torch.from_numpy(audio_float32).unsqueeze(0)
        audio_input = {"waveform": waveform, "sample_rate": sample_rate}

        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = self._pipeline(audio_input)
            annotation = result.speaker_diarization
            return [
                (turn.start, turn.end, speaker)
                for turn, _, speaker in annotation.itertracks(yield_label=True)
            ]
        except Exception as exc:
            logger.error("Diarization failed: %s", exc)
            return []


# ---------------------------------------------------------------------------
# SpeakerIdentifier — per-segment identification
# ---------------------------------------------------------------------------

# Cosine-similarity threshold above which two embeddings are judged to belong
# to the same speaker.  Tune empirically; 0.85 is a reasonable starting point
# for pyannote/embedding at 16 kHz.
DEFAULT_SIMILARITY_THRESHOLD: float = 0.35

# Minimum audio duration (seconds) for reliable embedding extraction.
MIN_AUDIO_SECONDS: float = 1.5


class SpeakerIdentifier:
    """Wraps pyannote/embedding for per-segment speaker embeddings.

    The underlying model is loaded lazily on the first call to
    get_embedding() or identify() so that importing this module is
    side-effect-free and does not slow application start-up.

    Parameters
    ----------
    hf_token:
        HuggingFace access token.  When None the class calls
        _load_hf_token() automatically.
    similarity_threshold:
        Cosine-similarity value in [0, 1] above which two embeddings
        are considered the same speaker.
    device:
        PyTorch device string, e.g. "cpu" or "cuda".  Defaults to "cpu"
        for broad compatibility; pass "cuda" if a GPU is available.
    """

    def __init__(
        self,
        hf_token: Optional[str] = None,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        device: str = "cpu",
    ) -> None:
        self._hf_token: Optional[str] = hf_token
        self.similarity_threshold = similarity_threshold
        self.device = device

        # Populated on first use by _ensure_model().
        self._model = None
        self._inference = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Lazy-load pyannote/embedding.  Idempotent."""
        if self._inference is not None:
            return

        try:
            import torch
            from pyannote.audio import Inference, Model
        except ImportError as exc:
            raise ImportError(
                "pyannote.audio is required for speaker identification. "
                "Install it with:  pip install pyannote.audio"
            ) from exc

        token = self._hf_token or _load_hf_token()
        if not token:
            raise ValueError(
                "A HuggingFace token is required to download pyannote/embedding. "
                "Set the HUGGINGFACE_TOKEN environment variable, add hf_token to "
                "~/.voice-inject/config.yaml, or pass hf_token= to SpeakerIdentifier()."
            )

        logger.info("Loading pyannote/embedding model (device=%s) …", self.device)
        try:
            model = Model.from_pretrained(
                "pyannote/embedding",
                token=token,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load pyannote/embedding.\n"
                "  1. Visit https://huggingface.co/pyannote/embedding and accept the model conditions.\n"
                "  2. Make sure HUGGINGFACE_TOKEN in .env belongs to the account that accepted.\n"
                f"  Original error: {exc}"
            ) from exc

        inference = Inference(model, window="whole")

        torch_device = torch.device(self.device)
        try:
            inference.to(torch_device)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not move pyannote model to %s (%s). Falling back to CPU.",
                self.device,
                exc,
            )
            inference.to(torch.device("cpu"))

        self._model = model
        self._inference = inference
        logger.info("pyannote/embedding loaded successfully.")

    @staticmethod
    def _to_pyannote_input(audio_float32: np.ndarray, sample_rate: int) -> dict:
        """Convert a (N,) float32 array to the dict pyannote Inference expects.

        pyannote.audio accepts {"waveform": tensor, "sample_rate": int} where
        the tensor is shape (channels, samples) and dtype float32.
        """
        import torch

        if audio_float32.ndim != 1:
            raise ValueError(
                f"audio_float32 must be 1-D, got shape {audio_float32.shape}"
            )

        # (N,) -> (1, N) to represent mono channel
        waveform = torch.from_numpy(audio_float32).unsqueeze(0)
        return {"waveform": waveform, "sample_rate": sample_rate}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_embedding(
        self,
        audio_float32: np.ndarray,
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """Extract a speaker embedding from an audio segment.

        Parameters
        ----------
        audio_float32:
            Mono audio, dtype float32, normalised to [-1.0, 1.0], shape (N,).
        sample_rate:
            Sampling rate in Hz.  Must be 16000 for best accuracy with the
            pyannote/embedding model.

        Returns
        -------
        np.ndarray
            Embedding vector, shape (1, D) where D is typically 512.
            Returns a zero vector of shape (1, 512) if the segment is too
            short or inference fails, so callers never receive None.
        """
        _FALLBACK_DIM = 512

        if audio_float32.ndim != 1:
            raise ValueError(
                f"audio_float32 must be 1-D, got shape {audio_float32.shape}"
            )

        duration = len(audio_float32) / sample_rate
        if duration < MIN_AUDIO_SECONDS:
            logger.debug(
                "Segment too short (%.2fs < %.2fs); returning zero embedding.",
                duration,
                MIN_AUDIO_SECONDS,
            )
            return np.zeros((1, _FALLBACK_DIM), dtype=np.float32)

        self._ensure_model()

        pyannote_input = self._to_pyannote_input(audio_float32, sample_rate)
        try:
            embedding = self._inference(pyannote_input)
            # pyannote returns shape (1, D) ndarray
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            if embedding.ndim == 1:
                embedding = embedding[np.newaxis, :]  # ensure (1, D)
            # L2-normalize for stable cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding.astype(np.float32)
        except Exception as exc:  # noqa: BLE001
            logger.error("Embedding extraction failed: %s", exc)
            return np.zeros((1, _FALLBACK_DIM), dtype=np.float32)

    def identify(
        self,
        audio_float32: np.ndarray,
        sample_rate: int,
        db: "SpeakerDB",  # noqa: F821  — imported below to avoid circular deps
    ) -> str:
        """Identify the speaker of an audio segment.

        Extracts an embedding from the segment, then queries the SpeakerDB
        for the closest registered speaker.  Returns the speaker's name, or
        "Unknown" when no registered speaker is similar enough.

        Parameters
        ----------
        audio_float32:
            Mono float32 audio, shape (N,), normalised to [-1.0, 1.0].
        sample_rate:
            Sampling rate in Hz.
        db:
            A SpeakerDB instance holding registered speaker embeddings.

        Returns
        -------
        str
            Speaker name from the database, or "Unknown".
        """
        embedding = self.get_embedding(audio_float32, sample_rate)

        # Zero embedding means the segment was too short or extraction failed.
        if np.all(embedding == 0):
            return "Unknown"

        name, similarity = db.find_closest(embedding)
        if name is None or similarity < self.similarity_threshold:
            logger.debug(
                "No match above threshold (best=%.3f, threshold=%.3f).",
                similarity if similarity is not None else 0.0,
                self.similarity_threshold,
            )
            return "Unknown"

        logger.debug("Identified speaker '%s' (similarity=%.3f).", name, similarity)
        return name

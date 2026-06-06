"""
speaker_db.py — Persistent speaker embedding database for voice-inject.

Stores name -> list-of-embeddings mappings in a JSON file at
~/.voice-inject/speakers.db. Embeddings are numpy arrays (typically
192- or 512-dim floats from pyannote/embedding) serialised as plain
Python lists for portability; no pickle required.

Cosine similarity is computed via scipy.spatial.distance.cdist, matching
the canonical pyannote pattern. Each speaker may hold multiple enrollment
embeddings; identification returns the best (maximum) similarity across
all stored embeddings for every registered speaker.

Usage
-----
    from speaker_db import SpeakerDB
    import numpy as np

    db = SpeakerDB()
    db.add_speaker("Alice", embedding_ndarray)   # enroll
    name = db.identify_speaker(query_embedding)  # "Alice" or None
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path.home() / ".voice-inject" / "speakers.db"


class SpeakerDB:
    """Persistent speaker-name-to-embedding store.

    Parameters
    ----------
    db_path:
        Path to the JSON database file.  Defaults to
        ``~/.voice-inject/speakers.db``.  The parent directory is
        created automatically if it does not exist.
    """

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        self._path = Path(db_path) if db_path is not None else _DEFAULT_DB_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Internal store: { name: [embedding_list, ...] }
        self._data: dict[str, list[list[float]]] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load the database from disk (no-op if file does not exist)."""
        if not self._path.exists():
            logger.debug("Speaker DB not found at %s — starting empty.", self._path)
            return
        try:
            with self._path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object at top level.")
            self._data = payload
            total = sum(len(v) for v in self._data.values())
            logger.debug(
                "Loaded speaker DB: %d speaker(s), %d embedding(s) from %s.",
                len(self._data),
                total,
                self._path,
            )
            # Normalize any unnormalized embeddings from older DB versions
            self._normalize_existing()
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "Could not parse speaker DB at %s (%s) — starting empty.", self._path, exc
            )
            self._data = {}

    def _normalize_existing(self) -> None:
        """L2-normalize any stored embeddings that aren't already unit-length."""
        modified = False
        for name, embeddings in self._data.items():
            for i, emb_list in enumerate(embeddings):
                arr = np.array(emb_list, dtype=np.float32)
                norm = np.linalg.norm(arr)
                if norm > 1.01:  # not yet normalized
                    arr = arr / norm
                    self._data[name][i] = arr.tolist()
                    modified = True
        if modified:
            logger.info("Normalized legacy embeddings in speaker DB.")
            self._save()

    def _save(self) -> None:
        """Write the current database to disk atomically."""
        tmp = self._path.with_suffix(".db.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2)
        tmp.replace(self._path)
        logger.debug("Saved speaker DB to %s.", self._path)

    def reload(self) -> None:
        """Re-read the database from disk — picks up speakers added by other processes."""
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_speaker(self, name: str, embedding: np.ndarray) -> None:
        """Enroll *embedding* under *name*.

        A speaker may be enrolled multiple times (e.g. from different
        audio clips); all embeddings are kept and used during
        identification.

        Parameters
        ----------
        name:
            Human-readable speaker name (case-sensitive).
        embedding:
            1-D or 2-D numpy array.  If 2-D with shape ``(1, D)`` (the
            default output of ``pyannote.audio.Inference``), the single
            row is stored.  Any other 2-D shape raises ``ValueError``.
        """
        if not name or not name.strip():
            raise ValueError("Speaker name must be a non-empty string.")

        vec = _to_1d(embedding)
        # L2-normalize before storing
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        entry = vec.tolist()

        if name not in self._data:
            self._data[name] = []
        self._data[name].append(entry)
        logger.info(
            "Enrolled embedding for '%s' (dim=%d, total=%d).",
            name,
            len(entry),
            len(self._data[name]),
        )
        self._save()

    MAX_EMBEDDINGS_PER_SPEAKER = 15

    def replace_speaker(self, name: str, embedding: np.ndarray) -> None:
        """Add a high-quality embedding for a speaker, keeping up to MAX_EMBEDDINGS_PER_SPEAKER.

        Older embeddings are dropped when the cap is reached, so the centroid
        naturally adapts as more sessions provide fresher samples.
        """
        if not name or not name.strip():
            raise ValueError("Speaker name must be a non-empty string.")

        vec = _to_1d(embedding)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        entry = vec.tolist()

        if name not in self._data:
            self._data[name] = []
        self._data[name].append(entry)
        # Keep only the most recent embeddings
        if len(self._data[name]) > self.MAX_EMBEDDINGS_PER_SPEAKER:
            self._data[name] = self._data[name][-self.MAX_EMBEDDINGS_PER_SPEAKER:]
        logger.info(
            "Enrolled embedding for '%s' (dim=%d, total=%d/%d).",
            name, len(entry), len(self._data[name]), self.MAX_EMBEDDINGS_PER_SPEAKER,
        )
        self._save()

    def identify_speaker(
        self,
        embedding: np.ndarray,
        threshold: float = 0.75,
    ) -> Optional[str]:
        """Return the best-matching speaker name or ``None``.

        For each registered speaker the maximum cosine similarity across
        all their stored embeddings is computed.  The speaker with the
        highest maximum similarity wins, provided it meets *threshold*.

        Parameters
        ----------
        embedding:
            Query embedding (1-D or 2-D ``(1, D)`` numpy array).
        threshold:
            Minimum cosine similarity required to declare a match.
            Values in ``[0, 1]``; 0.75 is a conservative default —
            tune upward (e.g. 0.85) to reduce false positives.

        Returns
        -------
        str | None
            Matched speaker name, or ``None`` when no speaker clears the
            threshold or the database is empty.
        """
        if not self._data:
            return None

        query = _to_1d(embedding).reshape(1, -1)  # shape (1, D)
        best_name: Optional[str] = None
        best_sim: float = -1.0

        for name, stored_lists in self._data.items():
            if not stored_lists:
                continue
            stored = np.array(stored_lists, dtype=np.float32)  # (N, D)
            centroid = stored.mean(axis=0, keepdims=True)  # (1, D)
            # Re-normalize centroid so cosine similarity remains in [-1, 1]
            c_norm = np.linalg.norm(centroid)
            if c_norm > 0:
                centroid = centroid / c_norm
            distance = cdist(query.astype(np.float32), centroid, metric="cosine")[0, 0]
            max_sim = float(1.0 - distance)
            logger.debug("  candidate='%s'  centroid_sim=%.4f", name, max_sim)
            if max_sim > best_sim:
                best_sim = max_sim
                best_name = name

        if best_sim >= threshold:
            logger.info("Identified speaker '%s' (similarity=%.4f).", best_name, best_sim)
            return best_name

        logger.debug(
            "No match above threshold %.2f (best was %.4f for '%s').",
            threshold,
            best_sim,
            best_name,
        )
        return None

    def find_closest(
        self,
        embedding: np.ndarray,
    ) -> tuple[Optional[str], Optional[float]]:
        """Return the best-matching speaker name and its cosine similarity.

        Unlike ``identify_speaker()``, this method does *not* apply a
        threshold — it simply returns the database entry with the highest
        similarity alongside that score, letting the caller decide whether
        to accept the match.  Returns ``(None, None)`` when the database is
        empty.

        Parameters
        ----------
        embedding:
            Query embedding (1-D or 2-D ``(1, D)`` numpy array).

        Returns
        -------
        tuple[str | None, float | None]
            ``(name, similarity)`` of the closest speaker, or
            ``(None, None)`` if the database is empty.
        """
        if not self._data:
            return None, None

        query = _to_1d(embedding).reshape(1, -1)  # shape (1, D)
        best_name: Optional[str] = None
        best_sim: float = -1.0

        for name, stored_lists in self._data.items():
            if not stored_lists:
                continue
            stored = np.array(stored_lists, dtype=np.float32)  # (N, D)
            centroid = stored.mean(axis=0, keepdims=True)  # (1, D)
            # Re-normalize centroid so cosine similarity remains in [-1, 1]
            c_norm = np.linalg.norm(centroid)
            if c_norm > 0:
                centroid = centroid / c_norm
            distance = cdist(query.astype(np.float32), centroid, metric="cosine")[0, 0]
            max_sim = float(1.0 - distance)
            if max_sim > best_sim:
                best_sim = max_sim
                best_name = name

        return best_name, best_sim if best_name is not None else None

    def list_speakers(self) -> list[str]:
        """Return a sorted list of all registered speaker names."""
        return sorted(self._data.keys())

    def remove_speaker(self, name: str) -> None:
        """Delete all embeddings for *name*.

        Parameters
        ----------
        name:
            Speaker name to remove.

        Raises
        ------
        KeyError
            If *name* is not in the database.
        """
        if name not in self._data:
            raise KeyError(f"Speaker '{name}' not found in database.")
        del self._data[name]
        logger.info("Removed speaker '%s' from database.", name)
        self._save()

    def get_embedding_count(self, name: str) -> int:
        """Return the number of stored embeddings for *name*.

        Parameters
        ----------
        name:
            Speaker name to query.

        Raises
        ------
        KeyError
            If *name* is not in the database.
        """
        if name not in self._data:
            raise KeyError(f"Speaker '{name}' not found in database.")
        return len(self._data[name])

    def __len__(self) -> int:
        """Return the number of registered speakers."""
        return len(self._data)

    def __repr__(self) -> str:  # pragma: no cover
        counts = {k: len(v) for k, v in self._data.items()}
        return f"SpeakerDB(path={self._path!r}, speakers={counts!r})"


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _to_1d(embedding: np.ndarray) -> np.ndarray:
    """Normalise a pyannote embedding to a flat 1-D float32 array.

    pyannote.audio.Inference returns shape ``(1, D)``.  We accept both
    ``(D,)`` and ``(1, D)``; anything else raises ``ValueError``.
    """
    arr = np.asarray(embedding, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[0] == 1:
        return arr[0]
    raise ValueError(
        f"Expected a 1-D embedding or a (1, D) array; got shape {arr.shape}."
    )

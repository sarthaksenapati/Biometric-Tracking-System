# db/models.py - Database models and operations

import numpy as np
import pickle
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from .connection import get_db


class Person:
    """Person model - represents a registered individual."""

    @staticmethod
    def create(name: str, display_name: Optional[str] = None, metadata: Optional[Dict] = None):
        """Create a new person."""
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO persons (name, display_name, metadata)
                       VALUES (%s, %s, %s)
                       ON CONFLICT (name) DO UPDATE
                       SET display_name = EXCLUDED.display_name,
                           updated_at = CURRENT_TIMESTAMP
                       RETURNING id""",
                    (name, display_name or name, json.dumps(metadata or {}))
                )
                result = cur.fetchone()
                return result['id'] if result else None

    @staticmethod
    def get_by_name(name: str):
        """Get person by name."""
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM persons WHERE name = %s", (name,))
                return cur.fetchone()

    @staticmethod
    def list_all():
        """List all persons."""
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT name, display_name, created_at FROM persons ORDER BY created_at")
                return cur.fetchall()

    @staticmethod
    def delete(name: str):
        """Delete a person and all their embeddings."""
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM persons WHERE name = %s", (name,))
                return cur.rowcount > 0


class Embedding:
    """Embedding model - stores face/body/gait embeddings in DB."""

    @staticmethod
    def save(person_name: str, modality: str, embedding: np.ndarray):
        """
        Save embedding(s) for a person.
        embedding can be:
          - shape (512,) - single embedding
          - shape (N, 512) - multiple exemplars
        """
        # Get person id
        person = Person.get_by_name(person_name)
        if not person:
            Person.create(person_name)

        person = Person.get_by_name(person_name)
        person_id = person['id']

        with get_db() as conn:
            with conn.cursor() as cur:
                # Handle multi-exemplar (shape N, 512)
                if embedding.ndim == 2:
                    # Delete existing exemplars for this modality
                    cur.execute(
                        "DELETE FROM embeddings WHERE person_id = %s AND modality = %s",
                        (person_id, modality)
                    )
                    # Insert each exemplar
                    for idx, emb in enumerate(embedding):
                        cur.execute(
                            """INSERT INTO embeddings (person_id, modality, embedding, shape, exemplar_index)
                               VALUES (%s, %s, %s, %s, %s)""",
                            (person_id, modality,
                             psycopg2.Binary(emb.tobytes()),
                             list(emb.shape), idx)
                        )
                else:
                    # Single embedding
                    cur.execute(
                        """INSERT INTO embeddings (person_id, modality, embedding, shape)
                           VALUES (%s, %s, %s, %s)
                           ON CONFLICT (person_id, modality, exemplar_index)
                           DO UPDATE SET embedding = EXCLUDED.embedding,
                                           shape = EXCLUDED.shape""",
                        (person_id, modality,
                         psycopg2.Binary(embedding.tobytes()),
                         list(embedding.shape))
                    )

    @staticmethod
    def load_all(modality: str) -> Dict[str, np.ndarray]:
        """
        Load all embeddings for a modality.
        Returns dict of person_name -> np.array
        (matches the interface of the old utils/embeddings.py)
        """
        result = {}
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT p.name, e.embedding, e.shape, e.exemplar_index
                    FROM embeddings e
                    JOIN persons p ON p.id = e.person_id
                    WHERE e.modality = %s
                    ORDER BY e.exemplar_index
                """, (modality,))

                rows = cur.fetchall()
                for row in rows:
                    name = row['name']
                    shape = tuple(row['shape'])
                    emb = np.frombuffer(row['embedding'], dtype=np.float32).reshape(shape)

                    if name not in result:
                        result[name] = emb
                    else:
                        # Multi-exemplar: stack
                        if result[name].ndim == 1:
                            result[name] = np.stack([result[name], emb])
                        else:
                            result[name] = np.vstack([result[name], emb])

        return result

    @staticmethod
    def load(person_name: str, modality: str) -> Optional[np.ndarray]:
        """Load embeddings for a specific person and modality."""
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT e.embedding, e.shape, e.exemplar_index
                    FROM embeddings e
                    JOIN persons p ON p.id = e.person_id
                    WHERE p.name = %s AND e.modality = %s
                    ORDER BY e.exemplar_index
                """, (person_name, modality))

                rows = cur.fetchall()
                if not rows:
                    return None

                embeddings = []
                for row in rows:
                    shape = tuple(row['shape'])
                    emb = np.frombuffer(row['embedding'], dtype=np.float32).reshape(shape)
                    embeddings.append(emb)

                if len(embeddings) == 1:
                    return embeddings[0]
                return np.stack(embeddings)


class Event:
    """Event model - stores sightings and handoffs."""

    @staticmethod
    def log(event_type: str, person_name: str, **kwargs):
        """Log an event."""
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO events (event_type, person_name, location, cam_id,
                                        confidence, elapsed_s, from_loc, to_loc,
                                        from_cam, to_cam, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    event_type, person_name,
                    kwargs.get('location'),
                    kwargs.get('cam_id'),
                    kwargs.get('confidence', 0.0),
                    kwargs.get('elapsed_s'),
                    kwargs.get('from_loc'),
                    kwargs.get('to_loc'),
                    kwargs.get('from_cam'),
                    kwargs.get('to_cam'),
                    kwargs.get('timestamp', datetime.now().timestamp())
                ))

    @staticmethod
    def get_recent(limit: int = 50, event_type: Optional[str] = None,
                   since: Optional[float] = None):
        """Get recent events."""
        query = "SELECT * FROM events WHERE 1=1"
        params = []

        if event_type:
            query += " AND event_type = %s"
            params.append(event_type)

        if since:
            query += " AND timestamp >= %s"
            params.append(since)

        query += " ORDER BY timestamp DESC LIMIT %s"
        params.append(limit)

        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.fetchall()

    @staticmethod
    def prune_old(older_than: float):
        """Delete events older than timestamp."""
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM events WHERE timestamp < %s", (older_than,))
                return cur.rowcount


class Camera:
    """Camera model - tracks camera status."""

    @staticmethod
    def update(cam_id: int, location: str, online: bool, source: str = None):
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO cameras (id, location, source, online, last_seen)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE
                    SET location = EXCLUDED.location,
                        source = EXCLUDED.source,
                        online = EXCLUDED.online,
                        last_seen = EXCLUDED.last_seen
                """, (cam_id, location, source, online, datetime.now().timestamp()))

    @staticmethod
    def get_all():
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM cameras ORDER BY id")
                return cur.fetchall()


class Detection:
    """Detection model - caches recent detections for fast lookup."""

    @staticmethod
    def save(person_name: str, cam_id: int, track_id: int,
              confidence: float, bbox: list, location: str, timestamp: float):
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO detections (person_name, cam_id, track_id,
                                            confidence, bbox, location, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (person_name, cam_id, track_id, confidence, bbox, location, timestamp))

    @staticmethod
    def get_active(cam_id: Optional[int] = None, since: Optional[float] = None):
        """Get active detections."""
        query = "SELECT * FROM detections WHERE 1=1"
        params = []

        if cam_id is not None:
            query += " AND cam_id = %s"
            params.append(cam_id)

        if since:
            query += " AND timestamp >= %s"
            params.append(since)

        query += " ORDER BY timestamp DESC"

        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.fetchall()

    @staticmethod
    def prune_old(older_than: float):
        """Delete old detections."""
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM detections WHERE timestamp < %s", (older_than,))
                return cur.rowcount

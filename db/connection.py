# db/connection.py - Database connection management

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager

# Database connection string from environment
# Format: postgresql://user:password@host:port/database
DB_URL = os.getenv("DATABASE_URL", "postgresql://biometric:biometric@localhost:5432/biometric_tracking")

# Global connection pool (simplified - use connection pooling in production)
_connection = None


def get_connection():
    """Get a database connection."""
    global _connection
    if _connection is None or _connection.closed:
        _connection = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
    return _connection


@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def init_db():
    """Initialize database schema."""
    with get_db() as conn:
        with conn.cursor() as cur:
            # Create persons table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS persons (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) UNIQUE NOT NULL,
                    display_name VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB DEFAULT '{}'::jsonb
                )
            """)

            # Create embeddings table (replaces .npy files)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id SERIAL PRIMARY KEY,
                    person_id INTEGER REFERENCES persons(id) ON DELETE CASCADE,
                    modality VARCHAR(50) NOT NULL,  -- 'face', 'body', 'gait'
                    embedding BYTEA NOT NULL,  -- numpy array stored as binary
                    shape INT[] NOT NULL,  -- original shape for reconstruction
                    exemplar_index INT DEFAULT 0,  -- for multi-exemplar
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(person_id, modality, exemplar_index)
                )
            """)

            # Create events table (replaces tracker_history.json)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id SERIAL PRIMARY KEY,
                    event_type VARCHAR(50) NOT NULL,  -- 'sighting', 'handoff'
                    person_name VARCHAR(255) NOT NULL,
                    location VARCHAR(255),
                    cam_id INT,
                    confidence FLOAT DEFAULT 0.0,
                    elapsed_s FLOAT,
                    from_loc VARCHAR(255),
                    to_loc VARCHAR(255),
                    from_cam INT,
                    to_cam INT,
                    timestamp FLOAT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create cameras table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS cameras (
                    id INT PRIMARY KEY,
                    location VARCHAR(255) NOT NULL,
                    source VARCHAR(500),
                    online BOOLEAN DEFAULT FALSE,
                    last_seen FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create detections table (for caching recent detections)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id SERIAL PRIMARY KEY,
                    person_name VARCHAR(255) NOT NULL,
                    cam_id INT NOT NULL,
                    track_id INT,
                    confidence FLOAT DEFAULT 0.0,
                    bbox INT[],  -- [x1, y1, x2, y2]
                    location VARCHAR(255),
                    timestamp FLOAT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for performance
            cur.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_person ON embeddings(person_id, modality)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_events_person ON events(person_name)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections(timestamp DESC)")

            print("[DB] ✅ Database schema initialized")


def close_db():
    """Close database connection."""
    global _connection
    if _connection and not _connection.closed:
        _connection.close()
        _connection = None

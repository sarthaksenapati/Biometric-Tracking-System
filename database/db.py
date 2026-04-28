import os
import io
import numpy as np
import psycopg2
from psycopg2.extras import DictCursor

# ── Connection ────────────────────────────────────────────────────────────────

def _get_db_url():
    return os.environ.get("DATABASE_URL")


def get_connection():
    url = _get_db_url()
    if url:
        return psycopg2.connect(url, cursor_factory=DictCursor)
    # Fallback to individual env vars
    return psycopg2.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=os.environ.get("DB_PORT", "5432"),
        database=os.environ.get("DB_NAME", "biometric_db"),
        user=os.environ.get("DB_USER", "postgres"),
        password=os.environ.get("DB_PASSWORD", ""),
        cursor_factory=DictCursor,
    )


def _db_available():
    try:
        conn = get_connection()
        conn.close()
        return True
    except Exception:
        return False


DB_AVAILABLE = _db_available()

# ── Schema ────────────────────────────────────────────────────────────────────

def init_db():
    sql = """
    CREATE TABLE IF NOT EXISTS persons (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL UNIQUE,
        display_name VARCHAR(255),
        created_at TIMESTAMP DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS embeddings (
        id SERIAL PRIMARY KEY,
        person_id INT REFERENCES persons(id) ON DELETE CASCADE,
        modality VARCHAR(50) NOT NULL,
        embedding BYTEA NOT NULL,
        shape VARCHAR(50),
        created_at TIMESTAMP DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_embeddings_person
        ON embeddings(person_id, modality);
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
        print("[DB] Tables initialized")
    finally:
        conn.close()


# ── Persons ──────────────────────────────────────────────────────────────────

def _get_person_id(conn, name):
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM persons WHERE name = %s", (name,))
        row = cur.fetchone()
        return row["id"] if row else None


def save_person(name, display_name=None):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO persons (name, display_name)
                   VALUES (%s, %s)
                   ON CONFLICT (name) DO UPDATE
                     SET display_name = EXCLUDED.display_name
                   RETURNING id""",
                (name, display_name or name),
            )
            pid = cur.fetchone()["id"]
        conn.commit()
        return pid
    finally:
        conn.close()


def rename_person(old_name, new_name):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE persons
                     SET name = %s, display_name = %s
                   WHERE name = %s""",
                (new_name, new_name, old_name),
            )
        conn.commit()
        print(f"[DB] Renamed person '{old_name}' → '{new_name}'")
    finally:
        conn.close()


def delete_person(name):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM persons WHERE name = %s", (name,))
        conn.commit()
    finally:
        conn.close()

# ── Embeddings ───────────────────────────────────────────────────────────────

def _serialize_emb(emb):
    buf = io.BytesIO()
    np.save(buf, emb)
    return buf.getvalue()


def _deserialize_emb(data):
    buf = io.BytesIO(data)
    return np.load(buf)


def save_embedding(person_name, modality, embedding):
    pid = save_person(person_name)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # Delete old embeddings of same person+modality
            cur.execute(
                "DELETE FROM embeddings WHERE person_id = %s AND modality = %s",
                (pid, modality),
            )
            cur.execute(
                """INSERT INTO embeddings (person_id, modality, embedding, shape)
                   VALUES (%s, %s, %s, %s)""",
                (pid, modality, _serialize_emb(embedding), str(embedding.shape)),
            )
        conn.commit()
        print(f"[DB] Saved {person_name}_{modality} shape={embedding.shape}")
    finally:
        conn.close()


def load_all_persons():
    """Returns {person_name: {modality: np.array, ...}}
       Same format as Matcher.database"""
    if not DB_AVAILABLE:
        return None
    conn = get_connection()
    try:
        result = {}
        with conn.cursor() as cur:
            cur.execute(
                """SELECT p.name, e.modality, e.embedding
                     FROM embeddings e
                     JOIN persons p ON p.id = e.person_id"""
            )
            for row in cur.fetchall():
                name = row["name"]
                mod = row["modality"]
                emb = _deserialize_emb(row["embedding"])
                if name not in result:
                    result[name] = {}
                result[name][mod] = emb
        return result
    except Exception as e:
        print(f"[DB] load_all_persons failed: {e}")
        return None
    finally:
        conn.close()


def load_all_embeddings(modality):
    """Returns {person_name: np.array} for given modality."""
    if not DB_AVAILABLE:
        return None
    conn = get_connection()
    try:
        result = {}
        with conn.cursor() as cur:
            cur.execute(
                """SELECT p.name, e.embedding
                     FROM embeddings e
                     JOIN persons p ON p.id = e.person_id
                    WHERE e.modality = %s""",
                (modality,),
            )
            for row in cur.fetchall():
                emb = _deserialize_emb(row["embedding"])
                result[row["name"]] = emb
        return result
    except Exception as e:
        print(f"[DB] load_all_embeddings failed: {e}")
        return None
    finally:
        conn.close()

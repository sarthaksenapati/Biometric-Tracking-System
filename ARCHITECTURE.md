# Architecture Improvements - Biometric Tracking System

## Overview

This document describes the architecture improvements made to the Biometric Tracking System:

1. **PostgreSQL Database** - Replaces .npy file storage
2. **Redis Caching** - For embeddings and recent detections
3. **Message Queue** - Async camera stream processing (optional)

---

## 1. PostgreSQL Database

### Why PostgreSQL?
- **Scalability**: Can handle thousands of persons/embeddings
- **Querying**: SQL queries for filtering, pagination, analytics
- **Persistence**: ACID compliance, backups, replication
- **Concurrency**: Multiple readers/writers without file locks

### Schema

```
persons
├── id SERIAL PRIMARY KEY
├── name VARCHAR(255) UNIQUE NOT NULL
├── display_name VARCHAR(255)
├── created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
├── updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
└── metadata JSONB DEFAULT '{}'::jsonb

embeddings
├── id SERIAL PRIMARY KEY
├── person_id INTEGER REFERENCES persons(id) ON DELETE CASCADE
├── modality VARCHAR(50) NOT NULL  -- 'face', 'body', 'gait'
├── embedding BYTEA NOT NULL
├── shape INT[] NOT NULL
├── exemplar_index INT DEFAULT 0
└── created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

events
├── id SERIAL PRIMARY KEY
├── event_type VARCHAR(50) NOT NULL  -- 'sighting', 'handoff'
├── person_name VARCHAR(255) NOT NULL
├── location VARCHAR(255)
├── cam_id INT
├── confidence FLOAT DEFAULT 0.0
├── elapsed_s FLOAT
├── from_loc VARCHAR(255)
├── to_loc VARCHAR(255)
├── timestamp FLOAT NOT NULL
└── created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

cameras
├── id INT PRIMARY KEY
├── location VARCHAR(255) NOT NULL
├── source VARCHAR(500)
├── online BOOLEAN DEFAULT FALSE
├── last_seen FLOAT
└── created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

detections
├── id SERIAL PRIMARY KEY
├── person_name VARCHAR(255) NOT NULL
├── cam_id INT NOT NULL
├── track_id INT
├── confidence FLOAT DEFAULT 0.0
├── bbox INT[]
├── location VARCHAR(255)
├── timestamp FLOAT NOT NULL
└── created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
```

### Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `db/__init__.py` | Created | Package init |
| `db/connection.py` | Created | PostgreSQL connection + schema init |
| `db/models.py` | Created | Person, Embedding, Event, Camera, Detection models |
| `core/matcher.py` | Modified | USE_DATABASE flag, load from DB + Redis cache |
| `utils/embeddings.py` | Modified | Save/load from DB with file fallback |
| `core/multi_tracker.py` | Modified | USE_DATABASE flag, events to DB |
| `run_tracker_multi.py` | Modified | DB initialization on startup |
| `dashboard.py` | Modified | Load history from DB |
| `requirements.txt` | Modified | Added psycopg2-binary |

### Usage

```bash
# Enable database usage
export USE_DATABASE=true
export DATABASE_URL=postgresql://biometric:biometric@localhost:5432/biometric_tracking

# Initialize database
python -c "from db.connection import init_db; init_db()"

# Run with database
python run_tracker_multi.py
```

### Fallback to File-Based Storage

If `USE_DATABASE=false` or database is unavailable, the system automatically falls back to:
- `.npy` files for embeddings
- `tracker_history.json` for event history

---

## 2. Redis Caching

### Why Redis?
- **Speed**: In-memory cache for frequent lookups
- **TTL**: Automatic expiration of cached data
- **Pub/Sub**: Can be used for real-time notifications

### What's Cached?

| Key Pattern | TTL | Description |
|-------------|-----|-------------|
| `emb:{name}:{modality}` | 1 hour | Person embedding |
| `det:{cam_id}` | 5 min | Recent detections |
| `identity:{name}` | 5 min | Cached identity for cross-camera |
| `tracker:state` | 10 sec | Cached tracker state for dashboard |

### Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `cache/__init__.py` | Created | Package init |
| `cache/redis_cache.py` | Created | Redis caching layer |
| `core/matcher.py` | Modified | Cache embeddings on load |
| `core/multi_tracker.py` | Modified | Cache detections + identity |
| `requirements.txt` | Modified | Added redis>=5.0.0 |

### Usage

```bash
# Enable Redis
export REDIS_URL=redis://localhost:6379

# Check Redis connection
python -c "from cache import get_cache; c=get_cache(); print(c.ping())"
```

---

## 3. Message Queue (Optional)

### Why Message Queue?
- **Async Processing**: Decouple frame capture from processing
- **Scalability**: Distribute processing across multiple workers
- **Reliability**: Messages persist until acknowledged

### Supported Backends

| Backend | Use Case |
|---------|----------|
| Redis Streams | Lightweight, already have Redis |
| RabbitMQ | Full-featured, persistence, routing |
| In-Memory | Development/testing (default fallback) |

### Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `queue/__init__.py` | Created | Package init |
| `queue/message_queue.py` | Created | Message queue abstraction |
| `core/multi_tracker.py` | Modified | USE_QUEUE flag |
| `requirements.txt` | Modified | Added pika>=1.3.0 (optional) |

### Usage

```bash
# Use Redis Streams (lightweight)
export USE_QUEUE=true
export QUEUE_TYPE=redis

# Use RabbitMQ (full-featured)
export QUEUE_TYPE=rabbitmq
export RABBITMQ_URL=amqp://guest:guest@localhost:5672/

# Publish frame (camera reader)
from queue import get_queue
q = get_queue()
q.publish_frame(cam_id, frame, detections, timestamp)

# Consume frames (processor)
def process_frame(message):
    cam_id = message['cam_id']
    frame = decode_frame(message['frame'])
    # process...

q.consume_frames(cam_id, process_frame)
```

---

## 4. Docker Services

### Updated docker-compose.yml

```yaml
services:
  postgres:    # New: PostgreSQL database
  redis:      # Existing: now also used by app
  tracker:     # Modified: uses DB + cache
  backend:     # Modified: uses DB
  dashboard:  # Modified: uses DB
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_DATABASE` | `true` | Enable PostgreSQL |
| `DATABASE_URL` | `postgresql://biometric:biometric@postgres:5432/biometric_tracking` | DB connection |
| `REDIS_URL` | `redis://redis:6379` | Redis connection |
| `USE_QUEUE` | `false` | Enable message queue |
| `QUEUE_TYPE` | `redis` | `redis` or `rabbitmq` |

---

## 5. Migration from File-Based Storage

### Automatic Migration

The system automatically handles migration:

1. **Matcher loads from DB first** → falls back to `.npy` files
2. **Embeddings saved to DB** → still saved to `.npy` as backup (optional)
3. **Events logged to DB** → still appended to `tracker_history.json` (optional)

### Manual Migration Script (Optional)

```python
# migrate_to_db.py
from db.models import Person, Embedding
from utils.embeddings import load_all_embeddings
import numpy as np

# Load all .npy embeddings
for modality in ['face', 'body', 'gait']:
    embeddings = load_all_embeddings(modality)
    for name, emb in embeddings.items():
        print(f"Migrating {name}_{modality}...")
        Person.create(name)
        Embedding.save(name, modality, emb)

print("Migration complete!")
```

---

## 6. Performance Comparison

| Operation | File-Based | PostgreSQL + Redis |
|-----------|------------|-------------------|
| Load all embeddings | O(N) file reads | O(1) cache hit |
| Similarity search | O(N) cosine sims | O(N) but cached |
| Save embedding | File write | DB insert + cache update |
| Query events | Parse JSON | SQL query with index |
| Concurrent access | File locks | MVCC (no locks) |

---

## 7. Authors

**Prityanshu Yadav** - Project Lead, Core Development  
**Sarthak Senapati** - Co-Developer, Architecture Improvements

---

## 8. Next Steps

1. **Test the new architecture**:
   ```bash
   docker-compose up -d postgres redis
   python -c "from db.connection import init_db; init_db()"
   python run_tracker_multi.py
   ```

2. **Monitor performance**:
   ```bash
   # Check DB
   docker-compose exec postgres psql -U biometric -d biometric_tracking -c "SELECT COUNT(*) FROM embeddings;"
   
   # Check Redis
   docker-compose exec redis redis-cli INFO keyspace
   ```

3. **Scale with message queue** (when needed):
   ```bash
   export USE_QUEUE=true
   docker-compose -f docker-compose.yml -f deploy/cloud/docker-compose.cloud.yml up -d
   ```

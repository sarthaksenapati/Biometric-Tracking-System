#!/usr/bin/env python3
"""
Migration script: Move embeddings from .npy files to PostgreSQL.

Usage:
    export USE_DATABASE=true
    export DATABASE_URL=postgresql://biometric:biometric@localhost:5432/biometric_tracking
    python migrate_to_db.py
"""

import os
import sys
import numpy as np

# Ensure database mode is on
os.environ['USE_DATABASE'] = 'true'

from db.connection import get_db
from db.models import Person, Embedding
from cache import get_cache


def migrate_embeddings():
    """Migrate all embeddings from .npy files to PostgreSQL."""
    print("=" * 60)
    print("Migration: .npy files → PostgreSQL")
    print("=" * 60)

    db_path = "embeddings_db"
    if not os.path.exists(db_path):
        print(f"❌ Folder not found: {db_path}")
        return False

    # Initialize database
    try:
        from db.connection import init_db
        init_db()
        print("✅ Database initialized")
    except Exception as e:
        print(f"❌ Database init failed: {e}")
        return False

    cache = get_cache()

    # Migrate each modality
    for modality in ["face", "body", "gait"]:
        print(f"\n── Migrating {modality} embeddings ──")

        files = [f for f in os.listdir(db_path) if f.endswith(f"_{modality}.npy")]

        for file in files:
            person_name = file.replace(f"_{modality}.npy", "")
            try:
                emb = np.load(os.path.join(db_path, file))
                print(f"  Processing {person_name}_{modality}.npy shape={emb.shape}...")

                # Create person if not exists
                Person.create(person_name)

                # Save embedding
                Embedding.save(person_name, modality, emb)

                # Cache in Redis
                if cache.is_available:
                    cache.cache_embedding(person_name, modality, emb)

                print(f"  ✅ Migrated {person_name}_{modality}")

            except Exception as e:
                print(f"  ❌ Failed {file}: {e}")

    print(f"\n{"=" * 60}")
    print("Migration complete!")
    print(f"{"=" * 60}")
    return True


if __name__ == "__main__":
    success = migrate_embeddings()
    sys.exit(0 if success else 1)

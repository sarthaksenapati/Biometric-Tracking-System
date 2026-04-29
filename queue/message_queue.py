# queue/message_queue.py - Message queue for async camera stream processing

import os
import json
import pickle
import base64
import numpy as np
from typing import Optional, Callable, Any, Dict
import threading
import time

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import pika  # RabbitMQ
    RABBITMQ_AVAILABLE = True
except ImportError:
    RABBITMQ_AVAILABLE = False

# Configuration
QUEUE_TYPE = os.getenv("QUEUE_TYPE", "redis")  # "redis" or "rabbitmq"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")

# Global queue instance
_queue_instance = None


def get_queue() -> 'MessageQueue':
    """Get or create the global queue instance."""
    global _queue_instance
    if _queue_instance is None:
        _queue_instance = MessageQueue()
    return _queue_instance


class MessageQueue:
    """
    Message queue abstraction for handling camera frames asynchronously.
    Supports Redis Streams and RabbitMQ.
    """

    def __init__(self, queue_type: Optional[str] = None):
        self.queue_type = queue_type or QUEUE_TYPE
        self._redis_client = None
        self._rabbit_connection = None
        self._rabbit_channel = None
        self._consumers = {}
        self._running = False

        if self.queue_type == "redis" and REDIS_AVAILABLE:
            try:
                self._redis_client = redis.from_url(REDIS_URL, decode_responses=False)
                self._redis_client.ping()
                print(f"[QUEUE] ✅ Using Redis Streams at {REDIS_URL}")
            except Exception as e:
                print(f"[QUEUE] ⚠️  Redis unavailable: {e}. Using in-memory queue.")
                self.queue_type = "memory"

        elif self.queue_type == "rabbitmq" and RABBITMQ_AVAILABLE:
            try:
                self._rabbit_connection = pika.BlockingConnection(
                    pika.URLParameters(RABBITMQ_URL)
                )
                self._rabbit_channel = self._rabbit_connection.channel()
                print(f"[QUEUE] ✅ Using RabbitMQ at {RABBITMQ_URL}")
            except Exception as e:
                print(f"[QUEUE] ⚠️  RabbitMQ unavailable: {e}. Using in-memory queue.")
                self.queue_type = "memory"

        else:
            print(f"[QUEUE] Using in-memory queue (type={self.queue_type})")
            self.queue_type = "memory"

        # In-memory fallback
        self._memory_queues = {}
        self._memory_locks = {}

    # ── Frame Publishing (Camera → Queue) ──────────────────────

    def publish_frame(self, cam_id: int, frame: np.ndarray,
                     detections: list, timestamp: Optional[float] = None):
        """
        Publish a camera frame for async processing.
        Frame is encoded as JPEG to save space.
        """
        timestamp = timestamp or time.time()

        # Encode frame as JPEG
        import cv2
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')

        message = {
            'cam_id': cam_id,
            'frame': frame_b64,
            'detections': detections,
            'timestamp': timestamp,
        }

        if self.queue_type == "redis":
            return self._publish_redis_frame(cam_id, message)
        elif self.queue_type == "rabbitmq":
            return self._publish_rabbitmq_frame(cam_id, message)
        else:
            return self._publish_memory_frame(cam_id, message)

    def _publish_redis_frame(self, cam_id: int, message: dict):
        """Publish frame to Redis Stream."""
        try:
            stream_key = f"camera:{cam_id}:frames"
            self._redis_client.xadd(
                stream_key,
                {'data': pickle.dumps(message)},
                maxlen=100,  # Keep last 100 frames
            )
            return True
        except Exception as e:
            print(f"[QUEUE] Redis publish error: {e}")
            return False

    def _publish_rabbitmq_frame(self, cam_id: int, message: dict):
        """Publish frame to RabbitMQ."""
        try:
            queue_name = f"camera_{cam_id}"
            self._rabbit_channel.queue_declare(queue=queue_name, durable=True)
            self._rabbit_channel.basic_publish(
                exchange='',
                routing_key=queue_name,
                body=pickle.dumps(message),
                properties=pika.BasicProperties(delivery_mode=2),  # Persistent
            )
            return True
        except Exception as e:
            print(f"[QUEUE] RabbitMQ publish error: {e}")
            return False

    def _publish_memory_frame(self, cam_id: int, message: dict):
        """Publish frame to in-memory queue."""
        queue_key = f"camera:{cam_id}"
        if queue_key not in self._memory_queues:
            from collections import deque
            self._memory_queues[queue_key] = deque(maxlen=100)
            self._memory_locks[queue_key] = threading.Lock()

        with self._memory_locks[queue_key]:
            self._memory_queues[queue_key].append(message)
        return True

    # ── Frame Consumption (Queue → Processor) ─────────────────

    def consume_frames(self, cam_id: int, callback: Callable[[dict], Any],
                      blocking: bool = True):
        """
        Consume frames from the queue.
        callback receives the message dict and should process it.
        """
        self._running = True

        if self.queue_type == "redis":
            self._consume_redis(cam_id, callback)
        elif self.queue_type == "rabbitmq":
            self._consume_rabbitmq(cam_id, callback)
        else:
            self._consume_memory(cam_id, callback, blocking)

    def _consume_redis(self, cam_id: int, callback: Callable):
        """Consume from Redis Stream."""
        stream_key = f"camera:{cam_id}:frames"
        last_id = '$'  # Start from newest

        while self._running:
            try:
                # Read from stream
                messages = self._redis_client.xread(
                    {stream_key: last_id},
                    block=1000,  # 1s timeout
                    count=10,
                )

                for stream, msgs in messages:
                    for msg_id, fields in msgs:
                        last_id = msg_id
                        data = pickle.loads(fields[b'data'])
                        callback(data)

            except Exception as e:
                print(f"[QUEUE] Redis consume error: {e}")
                time.sleep(1)

    def _consume_rabbitmq(self, cam_id: int, callback: Callable):
        """Consume from RabbitMQ."""
        queue_name = f"camera_{cam_id}"

        def on_message(ch, method, properties, body):
            try:
                message = pickle.loads(body)
                callback(message)
                ch.basic_ack(method.delivery_tag)
            except Exception as e:
                print(f"[QUEUE] RabbitMQ callback error: {e}")
                ch.basic_nack(method.delivery_tag, requeue=True)

        try:
            self._rabbit_channel.basic_qos(prefetch_count=1)
            self._rabbit_channel.basic_consume(
                queue=queue_name,
                on_message_callback=on_message,
            )
            self._rabbit_channel.start_consuming()
        except Exception as e:
            print(f"[QUEUE] RabbitMQ consume error: {e}")

    def _consume_memory(self, cam_id: int, callback: Callable, blocking: bool):
        """Consume from in-memory queue."""
        queue_key = f"camera:{cam_id}"

        while self._running:
            if queue_key not in self._memory_queues:
                if not blocking:
                    return
                time.sleep(0.1)
                continue

            with self._memory_locks[queue_key]:
                if not self._memory_queues[queue_key]:
                    if not blocking:
                        return
                    time.sleep(0.1)
                    continue
                message = self._memory_queues[queue_key].popleft()

            callback(message)

    # ── Embedding Message (for async registration) ─────────────

    def publish_embedding_task(self, person_name: str, modality: str,
                              embedding: np.ndarray):
        """Publish an embedding to be saved async."""
        message = {
            'type': 'save_embedding',
            'person_name': person_name,
            'modality': modality,
            'embedding': embedding.tobytes(),
            'shape': list(embedding.shape),
        }

        if self.queue_type == "redis":
            try:
                self._redis_client.lpush("embedding_tasks", pickle.dumps(message))
                return True
            except Exception as e:
                print(f"[QUEUE] Failed to publish embedding task: {e}")
                return False
        else:
            # In-memory: directly save (simplified)
            return False

    # ── Utility ─────────────────────────────────────────────────

    def get_queue_length(self, cam_id: int) -> int:
        """Get approximate queue length."""
        if self.queue_type == "redis":
            try:
                stream_key = f"camera:{cam_id}:frames"
                return self._redis_client.xlen(stream_key)
            except Exception:
                return 0
        else:
            queue_key = f"camera:{cam_id}"
            return len(self._memory_queues.get(queue_key, []))

    def stop(self):
        """Stop consuming."""
        self._running = False

    def close(self):
        """Close connections."""
        if self._redis_client:
            self._redis_client.close()
        if self._rabbit_connection and not self._rabbit_connection.is_closed:
            self._rabbit_connection.close()

"""
Redis Session Manager

Provides persistent session storage using Redis.
Meeting state survives server restarts.
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict
import redis.asyncio as redis
from dotenv import load_dotenv
import logging

load_dotenv(override=True)

logger = logging.getLogger(__name__)


@dataclass
class SerializableMeetingState:
    """Serializable version of meeting state for Redis storage"""
    user_id: str
    transcript: list
    full_transcript: str
    detected_questions: list
    suggestions: list
    last_processed_transcript_length: int
    last_activity: str  # ISO format string

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "SerializableMeetingState":
        return cls(**json.loads(data))


class RedisSessionManager:
    """
    Manages meeting sessions with Redis persistence.

    Sessions are stored in Redis with automatic expiry.
    """

    def __init__(self):
        self._redis: Optional[redis.Redis] = None
        self._connected = False
        self._session_ttl = 86400  # 24 hours

    def get_redis_url(self) -> Optional[str]:
        """Get Redis URL from environment"""
        return os.getenv("REDIS_URL")

    def is_configured(self) -> bool:
        """Check if Redis is configured"""
        return bool(self.get_redis_url())

    async def connect(self):
        """Connect to Redis"""
        redis_url = self.get_redis_url()
        if not redis_url:
            logger.info("Redis not configured - using in-memory sessions")
            return False

        try:
            self._redis = redis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            # Test connection
            await self._redis.ping()
            self._connected = True
            logger.info("✓ Connected to Redis")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._redis = None
            self._connected = False
            return False

    async def close(self):
        """Close Redis connection"""
        if self._redis:
            await self._redis.close()
            self._connected = False
            logger.info("Redis connection closed")

    def _session_key(self, user_id: str) -> str:
        """Generate Redis key for a session"""
        return f"meeting:session:{user_id}"

    async def save_session(self, user_id: str, state: "MeetingState"):
        """Save session state to Redis"""
        if not self._connected or not self._redis:
            return False

        try:
            serializable = SerializableMeetingState(
                user_id=state.user_id,
                transcript=state.transcript,
                full_transcript=state.full_transcript,
                detected_questions=state.detected_questions,
                suggestions=state.suggestions,
                last_processed_transcript_length=state.last_processed_transcript_length,
                last_activity=state.last_activity.isoformat()
            )

            await self._redis.setex(
                self._session_key(user_id),
                self._session_ttl,
                serializable.to_json()
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save session to Redis: {e}")
            return False

    async def load_session(self, user_id: str) -> Optional[dict]:
        """Load session state from Redis"""
        if not self._connected or not self._redis:
            return None

        try:
            data = await self._redis.get(self._session_key(user_id))
            if data:
                state = SerializableMeetingState.from_json(data)
                return {
                    "user_id": state.user_id,
                    "transcript": state.transcript,
                    "full_transcript": state.full_transcript,
                    "detected_questions": state.detected_questions,
                    "suggestions": state.suggestions,
                    "last_processed_transcript_length": state.last_processed_transcript_length,
                    "last_activity": datetime.fromisoformat(state.last_activity)
                }
            return None
        except Exception as e:
            logger.error(f"Failed to load session from Redis: {e}")
            return None

    async def delete_session(self, user_id: str) -> bool:
        """Delete a session from Redis"""
        if not self._connected or not self._redis:
            return False

        try:
            await self._redis.delete(self._session_key(user_id))
            return True
        except Exception as e:
            logger.error(f"Failed to delete session from Redis: {e}")
            return False

    async def list_sessions(self) -> list[str]:
        """List all active session user IDs"""
        if not self._connected or not self._redis:
            return []

        try:
            keys = await self._redis.keys("meeting:session:*")
            return [k.replace("meeting:session:", "") for k in keys]
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []

    async def extend_session(self, user_id: str) -> bool:
        """Extend session TTL"""
        if not self._connected or not self._redis:
            return False

        try:
            await self._redis.expire(self._session_key(user_id), self._session_ttl)
            return True
        except Exception as e:
            logger.error(f"Failed to extend session: {e}")
            return False


# Rate limiting with Redis
class RedisRateLimiter:
    """Rate limiter using Redis for distributed rate limiting"""

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self._redis = redis_client

    def set_redis(self, redis_client: redis.Redis):
        self._redis = redis_client

    async def check_rate_limit(
        self,
        user_id: str,
        limit: int,
        window_seconds: int,
        key_prefix: str = "ratelimit"
    ) -> tuple[bool, int]:
        """
        Check if user is within rate limit.

        Returns (is_allowed, remaining_requests)
        """
        if not self._redis:
            # No Redis - allow all
            return True, limit

        try:
            key = f"{key_prefix}:{user_id}:{window_seconds}"

            # Use Redis pipeline for atomic operations
            pipe = self._redis.pipeline()
            pipe.incr(key)
            pipe.expire(key, window_seconds)
            results = await pipe.execute()

            current_count = results[0]
            remaining = max(0, limit - current_count)
            is_allowed = current_count <= limit

            return is_allowed, remaining
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return True, limit  # Allow on error


# Global instances
redis_session_manager = RedisSessionManager()
redis_rate_limiter = RedisRateLimiter()

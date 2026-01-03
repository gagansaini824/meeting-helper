"""
Rate Limiter Module

Provides rate limiting functionality based on user ID and endpoint.
Uses a sliding window algorithm for accurate rate limiting.
"""

import asyncio
import time
from typing import Optional
from collections import defaultdict
from dataclasses import dataclass
from fastapi import HTTPException, Request


@dataclass
class RateLimitConfig:
    """Configuration for a rate limit"""
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int


# Default rate limits per tier
RATE_LIMITS = {
    "free": RateLimitConfig(
        requests_per_minute=20,
        requests_per_hour=200,
        requests_per_day=1000,
    ),
    "pro": RateLimitConfig(
        requests_per_minute=60,
        requests_per_hour=1000,
        requests_per_day=10000,
    ),
    "enterprise": RateLimitConfig(
        requests_per_minute=200,
        requests_per_hour=5000,
        requests_per_day=100000,
    ),
}


class RateLimiter:
    """
    Sliding window rate limiter.

    Tracks requests per user and enforces rate limits based on subscription tier.
    """

    def __init__(self):
        # Store request timestamps per user
        # Structure: {user_id: [timestamp1, timestamp2, ...]}
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    def _clean_old_requests(self, user_id: str, window_seconds: int) -> list[float]:
        """Remove requests older than the window"""
        cutoff = time.time() - window_seconds
        self._requests[user_id] = [
            ts for ts in self._requests[user_id]
            if ts > cutoff
        ]
        return self._requests[user_id]

    async def check_rate_limit(
        self,
        user_id: str,
        tier: str = "free",
        endpoint: Optional[str] = None
    ) -> tuple[bool, dict]:
        """
        Check if user is within rate limits.

        Returns:
            (is_allowed, rate_limit_info)
        """
        async with self._lock:
            config = RATE_LIMITS.get(tier, RATE_LIMITS["free"])
            now = time.time()

            # Clean old requests (keep last 24 hours)
            self._clean_old_requests(user_id, 86400)

            # Count requests in each window
            minute_ago = now - 60
            hour_ago = now - 3600
            day_ago = now - 86400

            requests = self._requests[user_id]
            minute_count = sum(1 for ts in requests if ts > minute_ago)
            hour_count = sum(1 for ts in requests if ts > hour_ago)
            day_count = len(requests)  # Already filtered to 24h

            # Check limits
            limit_info = {
                "minute": {
                    "used": minute_count,
                    "limit": config.requests_per_minute,
                    "remaining": max(0, config.requests_per_minute - minute_count)
                },
                "hour": {
                    "used": hour_count,
                    "limit": config.requests_per_hour,
                    "remaining": max(0, config.requests_per_hour - hour_count)
                },
                "day": {
                    "used": day_count,
                    "limit": config.requests_per_day,
                    "remaining": max(0, config.requests_per_day - day_count)
                }
            }

            # Check if any limit is exceeded
            is_allowed = (
                minute_count < config.requests_per_minute and
                hour_count < config.requests_per_hour and
                day_count < config.requests_per_day
            )

            return is_allowed, limit_info

    async def record_request(self, user_id: str):
        """Record a request for rate limiting"""
        async with self._lock:
            self._requests[user_id].append(time.time())

    async def get_rate_limit_headers(self, user_id: str, tier: str = "free") -> dict:
        """Get rate limit headers for response"""
        _, info = await self.check_rate_limit(user_id, tier)

        return {
            "X-RateLimit-Limit": str(info["minute"]["limit"]),
            "X-RateLimit-Remaining": str(info["minute"]["remaining"]),
            "X-RateLimit-Reset": str(int(time.time()) + 60),
        }


# Global rate limiter
rate_limiter = RateLimiter()


class RateLimitExceeded(HTTPException):
    """Exception raised when rate limit is exceeded"""
    def __init__(self, limit_info: dict):
        super().__init__(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "limits": limit_info,
                "message": "Too many requests. Please try again later."
            }
        )
        self.limit_info = limit_info


async def check_rate_limit_middleware(user_id: str, tier: str = "free"):
    """Middleware helper to check rate limits"""
    is_allowed, limit_info = await rate_limiter.check_rate_limit(user_id, tier)

    if not is_allowed:
        raise RateLimitExceeded(limit_info)

    await rate_limiter.record_request(user_id)
    return limit_info

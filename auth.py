"""
Clerk Authentication Module for FastAPI

This module provides JWT verification for Clerk authentication tokens.
It verifies tokens using Clerk's JWKS endpoint and extracts user information.
"""

import os
import httpx
import jwt
from jwt import PyJWKClient
from typing import Optional, Dict, Any
from dataclasses import dataclass
from functools import lru_cache
from fastapi import HTTPException, Request, WebSocket, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
import logging
import asyncio

# Load environment variables early
load_dotenv(override=True)

logger = logging.getLogger(__name__)

# Cache for user profiles to avoid repeated API calls
_user_profile_cache: Dict[str, tuple] = {}  # user_id -> (profile, timestamp)
USER_CACHE_TTL = 300  # 5 minutes

@dataclass
class ClerkUser:
    """Represents an authenticated Clerk user"""
    user_id: str
    email: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    image_url: Optional[str] = None

    @property
    def full_name(self) -> str:
        parts = [self.first_name, self.last_name]
        return " ".join(p for p in parts if p) or "User"


class ClerkAuth:
    """
    Clerk authentication handler for FastAPI.

    Verifies JWT tokens issued by Clerk and extracts user information.
    Uses JWKS for key validation.
    """

    def __init__(self):
        # Support both CLERK_PUBLISHABLE_KEY and NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY
        self.publishable_key = os.getenv("CLERK_PUBLISHABLE_KEY") or os.getenv("NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY", "")
        self.secret_key = os.getenv("CLERK_SECRET_KEY", "")
        self._jwks_client: Optional[PyJWKClient] = None
        self._frontend_api: Optional[str] = None

        if self.publishable_key:
            logger.info(f"Clerk configured with publishable key: {self.publishable_key[:20]}...")
        else:
            logger.warning("Clerk publishable key not found")

    @property
    def frontend_api(self) -> str:
        """Extract frontend API URL from publishable key"""
        if self._frontend_api is None:
            # Publishable key format: pk_test_xxx or pk_live_xxx
            # The frontend API is derived from the key
            if self.publishable_key:
                # Extract the frontend API from the key
                # Format: pk_test_<base64_encoded_frontend_api>
                try:
                    import base64
                    parts = self.publishable_key.split("_")
                    if len(parts) >= 3:
                        encoded = parts[2]
                        # Add padding if needed
                        padding = 4 - len(encoded) % 4
                        if padding != 4:
                            encoded += "=" * padding
                        decoded = base64.b64decode(encoded).decode('utf-8')
                        # Remove trailing $ if present
                        self._frontend_api = decoded.rstrip('$')
                    else:
                        self._frontend_api = ""
                except Exception as e:
                    logger.error(f"Failed to extract frontend API: {e}")
                    self._frontend_api = ""
            else:
                self._frontend_api = ""
        return self._frontend_api

    @property
    def jwks_url(self) -> str:
        """Get the JWKS URL for Clerk"""
        if self.frontend_api:
            return f"https://{self.frontend_api}/.well-known/jwks.json"
        return ""

    @property
    def jwks_client(self) -> Optional[PyJWKClient]:
        """Get or create JWKS client"""
        if self._jwks_client is None and self.jwks_url:
            try:
                # Set a short cache lifetime to handle key rotation
                self._jwks_client = PyJWKClient(self.jwks_url, cache_keys=True, lifespan=300)
            except Exception as e:
                logger.error(f"Failed to create JWKS client: {e}")
        return self._jwks_client

    def refresh_jwks(self):
        """Force refresh the JWKS client"""
        self._jwks_client = None
        return self.jwks_client

    def is_configured(self) -> bool:
        """Check if Clerk is properly configured"""
        return bool(self.publishable_key and self.secret_key)

    async def fetch_user_profile(self, user_id: str) -> Optional[dict]:
        """
        Fetch full user profile from Clerk's Backend API.

        The JWT only contains user_id (sub claim), so we need to call
        Clerk's API to get email, name, and image_url.
        """
        import time

        # Check cache first
        if user_id in _user_profile_cache:
            profile, timestamp = _user_profile_cache[user_id]
            if time.time() - timestamp < USER_CACHE_TTL:
                logger.debug(f"Using cached profile for user {user_id}")
                return profile

        if not self.secret_key:
            logger.warning("Clerk secret key not configured, cannot fetch user profile")
            return None

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"https://api.clerk.com/v1/users/{user_id}",
                    headers={
                        "Authorization": f"Bearer {self.secret_key}",
                        "Content-Type": "application/json"
                    },
                    timeout=10.0
                )

                if response.status_code == 200:
                    data = response.json()

                    # Extract primary email
                    email = None
                    email_addresses = data.get("email_addresses", [])
                    for email_obj in email_addresses:
                        if email_obj.get("id") == data.get("primary_email_address_id"):
                            email = email_obj.get("email_address")
                            break
                    # Fallback to first email if no primary found
                    if not email and email_addresses:
                        email = email_addresses[0].get("email_address")

                    profile = {
                        "user_id": user_id,
                        "email": email,
                        "first_name": data.get("first_name"),
                        "last_name": data.get("last_name"),
                        "image_url": data.get("image_url")
                    }

                    # Cache the profile
                    _user_profile_cache[user_id] = (profile, time.time())
                    logger.info(f"Fetched profile for user {user_id}: {email}")

                    return profile
                else:
                    logger.error(f"Failed to fetch user profile: {response.status_code} - {response.text}")
                    return None

        except Exception as e:
            logger.error(f"Error fetching user profile: {e}")
            return None

    async def verify_token(self, token: str) -> Optional[ClerkUser]:
        """
        Verify a Clerk JWT token and return user information.

        Args:
            token: The JWT token to verify

        Returns:
            ClerkUser if valid, None otherwise
        """
        if not self.is_configured():
            logger.warning("Clerk not configured, skipping authentication")
            return None

        if not token:
            return None

        try:
            # Get the signing key from JWKS
            if not self.jwks_client:
                logger.error("JWKS client not available")
                return None

            signing_key = self.jwks_client.get_signing_key_from_jwt(token)

            # Decode and verify the token
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                options={
                    "verify_aud": False,  # Clerk tokens don't always have audience
                    "verify_iss": True,
                }
            )

            # Extract user information from the token
            user_id = payload.get("sub")
            if not user_id:
                logger.error("Token missing 'sub' claim")
                return None

            # JWT might have some claims, but Clerk's session tokens typically
            # only have the user_id. Fetch full profile from Clerk API.
            email = payload.get("email")
            first_name = payload.get("first_name")
            last_name = payload.get("last_name")
            image_url = payload.get("image_url")

            # If essential fields are missing, fetch from Clerk API
            if not email:
                logger.info(f"JWT missing email for user {user_id}, fetching from Clerk API")
                profile = await self.fetch_user_profile(user_id)
                if profile:
                    email = profile.get("email")
                    first_name = profile.get("first_name") or first_name
                    last_name = profile.get("last_name") or last_name
                    image_url = profile.get("image_url") or image_url

            return ClerkUser(
                user_id=user_id,
                email=email,
                first_name=first_name,
                last_name=last_name,
                image_url=image_url
            )

        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            return None

    async def get_current_user(self, request: Request) -> ClerkUser:
        """
        Extract and verify the current user from a request.

        Looks for the token in:
        1. Authorization header (Bearer token)
        2. __session cookie (Clerk's default)

        Raises HTTPException if authentication fails.
        """
        token = None

        # Try Authorization header first
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]

        # Fall back to cookie
        if not token:
            token = request.cookies.get("__session")

        if not token:
            raise HTTPException(status_code=401, detail="Not authenticated")

        user = await self.verify_token(token)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid or expired token")

        return user

    async def get_optional_user(self, request: Request) -> Optional[ClerkUser]:
        """
        Try to get the current user, but return None instead of raising if not authenticated.
        Useful for endpoints that work differently for authenticated vs anonymous users.
        """
        try:
            return await self.get_current_user(request)
        except HTTPException:
            return None

    async def verify_websocket(self, websocket: WebSocket, token: str = Query(None)) -> Optional[ClerkUser]:
        """
        Verify authentication for WebSocket connections.

        Args:
            websocket: The WebSocket connection
            token: JWT token passed as query parameter

        Returns:
            ClerkUser if authenticated, None otherwise
        """
        if not token:
            # Try to get from cookies in the initial handshake
            token = websocket.cookies.get("__session")

        if not token:
            logger.debug("No token provided for WebSocket")
            return None

        user = await self.verify_token(token)
        if user:
            logger.info(f"WebSocket authenticated: {user.user_id} ({user.email})")
        else:
            logger.warning("WebSocket token verification failed")
        return user


# Singleton instance
clerk_auth = ClerkAuth()


# FastAPI dependency functions
async def require_auth(request: Request) -> ClerkUser:
    """Dependency that requires authentication"""
    return await clerk_auth.get_current_user(request)


async def optional_auth(request: Request) -> Optional[ClerkUser]:
    """Dependency that optionally authenticates"""
    return await clerk_auth.get_optional_user(request)

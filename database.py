"""
Database Models and Setup

This module provides database models using SQLAlchemy for user data,
documents, subscriptions, and usage tracking.

Supports both SQLite (development) and PostgreSQL (production).
PostgreSQL supports pgvector for efficient vector similarity search.
"""

import os
import json
from datetime import datetime
from typing import Optional, List
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from sqlalchemy import (
    Column, String, Text, DateTime, Integer, Float, Boolean,
    ForeignKey, JSON, create_engine, event, text
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import (
    sessionmaker, relationship, declarative_base
)
from sqlalchemy.pool import StaticPool

load_dotenv(override=True)

Base = declarative_base()

# Check if we're using PostgreSQL (for pgvector support)
_is_postgres = bool(os.getenv("DATABASE_URL"))


# ============= Models =============

class User(Base):
    """User profile synced from Clerk"""
    __tablename__ = "users"

    id = Column(String(255), primary_key=True)  # Clerk user ID
    email = Column(String(255), nullable=True)
    first_name = Column(String(255), nullable=True)
    last_name = Column(String(255), nullable=True)
    image_url = Column(Text, nullable=True)
    subscription_tier = Column(String(50), default="free")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    settings = relationship("UserSettings", back_populates="user", uselist=False, cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="owner", cascade="all, delete-orphan")
    usage_records = relationship("UsageRecord", back_populates="user", cascade="all, delete-orphan")
    meeting_sessions = relationship("MeetingSession", back_populates="user", cascade="all, delete-orphan")

    @property
    def full_name(self) -> str:
        parts = [self.first_name, self.last_name]
        return " ".join(p for p in parts if p) or "User"


class UserSettings(Base):
    """User settings and API keys"""
    __tablename__ = "user_settings"

    user_id = Column(String(255), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    max_documents = Column(Integer, default=10)
    max_meeting_minutes = Column(Integer, default=60)
    max_storage_mb = Column(Integer, default=100)

    # Encrypted API keys (optional - BYOK)
    deepgram_key_encrypted = Column(Text, nullable=True)
    openai_key_encrypted = Column(Text, nullable=True)
    anthropic_key_encrypted = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="settings")


class Document(Base):
    """User's uploaded documents"""
    __tablename__ = "documents"

    id = Column(String(255), primary_key=True)
    user_id = Column(String(255), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    content_type = Column(String(100), nullable=True)
    size_bytes = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    owner = relationship("User", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    """Document chunks with embeddings for semantic search"""
    __tablename__ = "document_chunks"

    id = Column(String(255), primary_key=True)
    document_id = Column(String(255), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(JSON, nullable=True)  # Store as JSON array for SQLite compatibility
    chunk_index = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    document = relationship("Document", back_populates="chunks")


class MeetingSession(Base):
    """User's meeting sessions with transcripts"""
    __tablename__ = "meeting_sessions"

    id = Column(String(255), primary_key=True)
    user_id = Column(String(255), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(500), nullable=True)
    transcript = Column(Text, default="")
    questions = Column(JSON, default=list)  # List of detected questions
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Integer, default=0)

    # Relationships
    user = relationship("User", back_populates="meeting_sessions")


class UsageRecord(Base):
    """Track API usage for billing/limits"""
    __tablename__ = "usage_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    service = Column(String(50), nullable=False)  # deepgram, openai, anthropic
    operation = Column(String(100), nullable=False)  # transcription, embedding, completion
    tokens = Column(Integer, default=0)
    cost_cents = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="usage_records")


class AuditLog(Base):
    """Audit log for compliance"""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=True)  # Can be null for system events
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50), nullable=True)
    resource_id = Column(String(255), nullable=True)
    extra_data = Column(JSON, nullable=True)  # Renamed from 'metadata' which is reserved
    ip_address = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# ============= Database Setup =============

class DatabaseManager:
    """
    Manages database connections and sessions.

    Supports SQLite for development and PostgreSQL for production.
    """

    def __init__(self):
        self._engine = None
        self._session_factory = None

    def get_database_url(self) -> str:
        """Get database URL from environment"""
        # Check for PostgreSQL URL first
        postgres_url = os.getenv("DATABASE_URL")
        if postgres_url:
            # Handle Heroku-style postgres:// URLs
            if postgres_url.startswith("postgres://"):
                postgres_url = postgres_url.replace("postgres://", "postgresql+asyncpg://", 1)
            elif postgres_url.startswith("postgresql://"):
                postgres_url = postgres_url.replace("postgresql://", "postgresql+asyncpg://", 1)
            return postgres_url

        # Fall back to SQLite
        sqlite_path = os.getenv("SQLITE_PATH", "data/meeting_assistant.db")
        return f"sqlite+aiosqlite:///{sqlite_path}"

    async def init_db(self):
        """Initialize database connection and create tables"""
        db_url = self.get_database_url()
        is_sqlite = db_url.startswith("sqlite")
        self._is_postgres = not is_sqlite

        if is_sqlite:
            # SQLite-specific settings
            self._engine = create_async_engine(
                db_url,
                echo=False,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool
            )
        else:
            # PostgreSQL settings
            self._engine = create_async_engine(
                db_url,
                echo=False,
                pool_size=5,
                max_overflow=10
            )

        self._session_factory = sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

        # For PostgreSQL, enable pgvector extension
        if self._is_postgres:
            async with self._engine.begin() as conn:
                try:
                    await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                    print("✓ pgvector extension enabled")
                except Exception as e:
                    print(f"Note: pgvector extension not available: {e}")

        # Create tables
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        print(f"Database initialized: {'SQLite' if is_sqlite else 'PostgreSQL'}")

    @property
    def is_postgres(self) -> bool:
        """Check if using PostgreSQL"""
        return getattr(self, '_is_postgres', False)

    async def close(self):
        """Close database connections"""
        if self._engine:
            await self._engine.dispose()

    @asynccontextmanager
    async def session(self):
        """Get a database session"""
        if not self._session_factory:
            await self.init_db()

        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise


# Global database manager
db = DatabaseManager()


# ============= Helper Functions =============

async def get_or_create_user(
    user_id: str,
    email: Optional[str] = None,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    image_url: Optional[str] = None
) -> User:
    """Get or create a user record, updating profile if new data is available"""
    from sqlalchemy import select

    async with db.session() as session:
        result = await session.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()

        if not user:
            print(f"Creating new user: {user_id} ({email})")
            user = User(
                id=user_id,
                email=email,
                first_name=first_name,
                last_name=last_name,
                image_url=image_url
            )
            session.add(user)

            # Create default settings
            settings = UserSettings(user_id=user_id)
            session.add(settings)

            await session.commit()
            print(f"✓ User {user_id} saved to database with email: {email}")
        else:
            # Update user profile if new data is available and current is NULL
            updated = False
            if email and not user.email:
                user.email = email
                updated = True
            if first_name and not user.first_name:
                user.first_name = first_name
                updated = True
            if last_name and not user.last_name:
                user.last_name = last_name
                updated = True
            if image_url and not user.image_url:
                user.image_url = image_url
                updated = True

            if updated:
                await session.commit()
                print(f"✓ Updated user {user_id} profile: email={email}, name={first_name} {last_name}")

        return user


async def log_usage(
    user_id: str,
    service: str,
    operation: str,
    tokens: int = 0,
    cost_cents: int = 0
):
    """Log API usage for a user"""
    async with db.session() as session:
        record = UsageRecord(
            user_id=user_id,
            service=service,
            operation=operation,
            tokens=tokens,
            cost_cents=cost_cents
        )
        session.add(record)
        await session.commit()


async def log_audit(
    action: str,
    user_id: Optional[str] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    extra_data: Optional[dict] = None,
    ip_address: Optional[str] = None
):
    """Log an audit event"""
    async with db.session() as session:
        log = AuditLog(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            extra_data=extra_data,
            ip_address=ip_address
        )
        session.add(log)
        await session.commit()


# ============= Vector Search Functions (PostgreSQL with pgvector) =============

async def search_similar_chunks(
    user_id: str,
    query_embedding: List[float],
    limit: int = 5
) -> List[dict]:
    """
    Search for similar document chunks using pgvector.
    Falls back to JSON-based search for SQLite.
    """
    from sqlalchemy import select
    import numpy as np

    async with db.session() as session:
        if db.is_postgres:
            # Use pgvector for efficient similarity search
            embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
            query = text("""
                SELECT dc.id, dc.content, dc.chunk_index, d.name as document_name, d.id as document_id,
                       dc.embedding <=> :embedding::vector as distance
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id
                WHERE d.user_id = :user_id AND dc.embedding IS NOT NULL
                ORDER BY dc.embedding <=> :embedding::vector
                LIMIT :limit
            """)
            result = await session.execute(
                query,
                {"user_id": user_id, "embedding": embedding_str, "limit": limit}
            )
            rows = result.fetchall()
            return [
                {
                    "id": row.id,
                    "content": row.content,
                    "chunk_index": row.chunk_index,
                    "document_name": row.document_name,
                    "document_id": row.document_id,
                    "distance": row.distance
                }
                for row in rows
            ]
        else:
            # SQLite fallback: load all chunks and compute similarity in Python
            query = select(DocumentChunk, Document).join(
                Document, DocumentChunk.document_id == Document.id
            ).where(Document.user_id == user_id)

            result = await session.execute(query)
            rows = result.fetchall()

            if not rows:
                return []

            # Compute cosine similarity
            query_vec = np.array(query_embedding)
            scored = []

            for chunk, doc in rows:
                if chunk.embedding:
                    chunk_vec = np.array(chunk.embedding)
                    # Cosine similarity
                    similarity = np.dot(query_vec, chunk_vec) / (
                        np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec) + 1e-8
                    )
                    scored.append({
                        "id": chunk.id,
                        "content": chunk.content,
                        "chunk_index": chunk.chunk_index,
                        "document_name": doc.name,
                        "document_id": doc.id,
                        "distance": 1 - similarity  # Convert to distance
                    })

            # Sort by distance (lower is better)
            scored.sort(key=lambda x: x["distance"])
            return scored[:limit]


async def save_document_to_db(
    user_id: str,
    doc_id: str,
    name: str,
    content: str,
    chunks: List[dict]
) -> Document:
    """Save a document and its chunks to the database"""
    from sqlalchemy import select
    import uuid

    async with db.session() as session:
        # Create or update document
        result = await session.execute(
            select(Document).where(Document.id == doc_id)
        )
        doc = result.scalar_one_or_none()

        if not doc:
            doc = Document(
                id=doc_id,
                user_id=user_id,
                name=name,
                content=content,
                size_bytes=len(content.encode('utf-8'))
            )
            session.add(doc)

        # Add chunks with embeddings
        for i, chunk_data in enumerate(chunks):
            chunk_id = str(uuid.uuid4())

            if db.is_postgres:
                # For PostgreSQL, store embedding as vector
                embedding = chunk_data.get("embedding")
                if embedding:
                    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
                    await session.execute(text("""
                        INSERT INTO document_chunks (id, document_id, content, embedding, chunk_index)
                        VALUES (:id, :doc_id, :content, :embedding::vector, :idx)
                    """), {
                        "id": chunk_id,
                        "doc_id": doc_id,
                        "content": chunk_data["content"],
                        "embedding": embedding_str,
                        "idx": i
                    })
            else:
                # For SQLite, store embedding as JSON
                chunk = DocumentChunk(
                    id=chunk_id,
                    document_id=doc_id,
                    content=chunk_data["content"],
                    embedding=chunk_data.get("embedding"),
                    chunk_index=i
                )
                session.add(chunk)

        await session.commit()
        return doc


async def get_user_documents(user_id: str) -> List[dict]:
    """Get all documents for a user"""
    from sqlalchemy import select, func

    async with db.session() as session:
        # Get documents with chunk counts
        query = select(
            Document,
            func.count(DocumentChunk.id).label("chunk_count")
        ).outerjoin(DocumentChunk).where(
            Document.user_id == user_id
        ).group_by(Document.id)

        result = await session.execute(query)
        rows = result.fetchall()

        return [
            {
                "id": doc.id,
                "name": doc.name,
                "created_at": doc.created_at.isoformat(),
                "size_bytes": doc.size_bytes,
                "chunk_count": chunk_count
            }
            for doc, chunk_count in rows
        ]


async def delete_document_from_db(user_id: str, doc_id: str) -> bool:
    """Delete a document and its chunks"""
    from sqlalchemy import select, delete

    async with db.session() as session:
        # Verify ownership
        result = await session.execute(
            select(Document).where(
                Document.id == doc_id,
                Document.user_id == user_id
            )
        )
        doc = result.scalar_one_or_none()

        if not doc:
            return False

        # Delete chunks first (cascade should handle this, but explicit is safer)
        await session.execute(
            delete(DocumentChunk).where(DocumentChunk.document_id == doc_id)
        )

        # Delete document
        await session.execute(
            delete(Document).where(Document.id == doc_id)
        )

        await session.commit()
        return True

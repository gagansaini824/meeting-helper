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
    """User's uploaded documents - metadata only, content stored in Pinecone"""
    __tablename__ = "documents"

    id = Column(String(255), primary_key=True)
    user_id = Column(String(255), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(500), nullable=False)
    file_type = Column(String(100), nullable=True)
    size_bytes = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    pinecone_namespace = Column(String(255), nullable=True)  # Usually user_id
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    owner = relationship("User", back_populates="documents")
    sessions = relationship("DocumentSession", back_populates="document", cascade="all, delete-orphan")


class DocumentSession(Base):
    """Junction table linking documents to sessions (many-to-many)"""
    __tablename__ = "document_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String(255), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    session_id = Column(String(255), ForeignKey("meeting_sessions.id", ondelete="CASCADE"), nullable=False)
    added_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    document = relationship("Document", back_populates="sessions")
    session = relationship("MeetingSession", back_populates="document_links")


class MeetingSession(Base):
    """User's interview/meeting sessions with transcripts, questions, and answers"""
    __tablename__ = "meeting_sessions"

    id = Column(String(255), primary_key=True)
    user_id = Column(String(255), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(500), nullable=True)
    status = Column(String(50), default="active")  # active, archived

    # JSONB fields for flexible data storage
    transcript_entries = Column(JSON, default=list)  # [{speaker: int, text: str, timestamp: str, is_final: bool}]
    detected_questions = Column(JSON, default=list)  # [{text: str, timestamp: str, answered: bool}]
    answers = Column(JSON, default=list)  # [{question: str, answer: str, timestamp: str, sources: []}]
    document_ids = Column(JSON, default=list)  # [doc_id, ...] - documents used in this session
    conversation_summaries = Column(JSON, default=list)  # [{timestamp: str, summary: str, transcript_length: int}]

    # Full transcript text for easy search
    full_transcript = Column(Text, default="")

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", back_populates="meeting_sessions")
    document_links = relationship("DocumentSession", back_populates="session", cascade="all, delete-orphan")


class UsageRecord(Base):
    """Track API usage for billing/limits"""
    __tablename__ = "usage_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    session_id = Column(String(255), ForeignKey("meeting_sessions.id", ondelete="SET NULL"), nullable=True)
    service = Column(String(50), nullable=False)  # deepgram, openai, anthropic
    operation = Column(String(100), nullable=False)  # transcription, embedding, completion
    tokens = Column(Integer, default=0)
    cost_cents = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="usage_records")
    session = relationship("MeetingSession")


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
        postgres_url = os.getenv("DATABASE_URL")
        if not postgres_url:
            raise ValueError(
                "DATABASE_URL environment variable is required. "
                "Please set it to your PostgreSQL connection string."
            )

        # Handle Heroku/Railway-style postgres:// URLs
        if postgres_url.startswith("postgres://"):
            postgres_url = postgres_url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif postgres_url.startswith("postgresql://"):
            postgres_url = postgres_url.replace("postgresql://", "postgresql+asyncpg://", 1)

        return postgres_url

    async def init_db(self):
        """Initialize database connection and create tables"""
        db_url = self.get_database_url()
        self._is_postgres = True

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


        # Create tables
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # Run migrations for meeting_sessions table
        await self._migrate_meeting_sessions()

        print("Database initialized: PostgreSQL")

    async def _migrate_meeting_sessions(self):
        """Add new columns to meeting_sessions if they don't exist"""
        migrations = [
            "ALTER TABLE meeting_sessions ADD COLUMN IF NOT EXISTS status VARCHAR(50) DEFAULT 'active'",
            "ALTER TABLE meeting_sessions ADD COLUMN IF NOT EXISTS transcript_entries JSON DEFAULT '[]'::json",
            "ALTER TABLE meeting_sessions ADD COLUMN IF NOT EXISTS detected_questions JSON DEFAULT '[]'::json",
            "ALTER TABLE meeting_sessions ADD COLUMN IF NOT EXISTS answers JSON DEFAULT '[]'::json",
            "ALTER TABLE meeting_sessions ADD COLUMN IF NOT EXISTS document_ids JSON DEFAULT '[]'::json",
            "ALTER TABLE meeting_sessions ADD COLUMN IF NOT EXISTS full_transcript TEXT DEFAULT ''",
            "ALTER TABLE meeting_sessions ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT NOW()",
            "ALTER TABLE meeting_sessions ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT NOW()",
            "ALTER TABLE meeting_sessions ADD COLUMN IF NOT EXISTS ended_at TIMESTAMP"
        ]

        async with self._engine.begin() as conn:
            for migration in migrations:
                try:
                    await conn.execute(text(migration))
                except Exception as e:
                    # Column might already exist or other issue, log but continue
                    print(f"Migration note: {e}")

        # Run schema optimization migrations
        await self._migrate_schema_v2()

    async def _migrate_schema_v2(self):
        """
        Schema optimization v2:
        - documents: remove content column, add chunk_count, pinecone_namespace, file_type
        - Drop document_chunks table (content stored in Pinecone only)
        - Add document_sessions junction table
        - Add session_id to usage_records
        """
        migrations = [
            # Documents table changes
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS file_type VARCHAR(100)",
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS chunk_count INTEGER DEFAULT 0",
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS pinecone_namespace VARCHAR(255)",

            # Drop the old content column if it exists (data now in Pinecone only)
            "ALTER TABLE documents DROP COLUMN IF EXISTS content",
            "ALTER TABLE documents DROP COLUMN IF EXISTS content_type",

            # Drop document_chunks table (all chunk data in Pinecone)
            "DROP TABLE IF EXISTS document_chunks CASCADE",

            # Create document_sessions junction table
            """CREATE TABLE IF NOT EXISTS document_sessions (
                id SERIAL PRIMARY KEY,
                document_id VARCHAR(255) NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                session_id VARCHAR(255) NOT NULL REFERENCES meeting_sessions(id) ON DELETE CASCADE,
                added_at TIMESTAMP DEFAULT NOW()
            )""",

            # Add index for efficient lookups
            "CREATE INDEX IF NOT EXISTS idx_document_sessions_document ON document_sessions(document_id)",
            "CREATE INDEX IF NOT EXISTS idx_document_sessions_session ON document_sessions(session_id)",

            # Add session_id to usage_records
            "ALTER TABLE usage_records ADD COLUMN IF NOT EXISTS session_id VARCHAR(255) REFERENCES meeting_sessions(id) ON DELETE SET NULL"
        ]

        async with self._engine.begin() as conn:
            for migration in migrations:
                try:
                    await conn.execute(text(migration))
                except Exception as e:
                    # Table/column might already exist or other issue
                    print(f"Schema v2 migration note: {e}")

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
    cost_cents: int = 0,
    session_id: Optional[str] = None
):
    """Log API usage for a user, optionally tied to a session"""
    async with db.session() as session:
        record = UsageRecord(
            user_id=user_id,
            session_id=session_id,
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


# ============= Document Management Functions =============
# Note: Vector search is handled by Pinecone (see vector_store.py)


async def save_document_to_db(
    user_id: str,
    doc_id: str,
    name: str,
    file_type: str,
    size_bytes: int,
    chunk_count: int,
    session_id: str,
    pinecone_namespace: Optional[str] = None
) -> Document:
    """
    Save document metadata to the database.
    Note: Document content and embeddings are stored in Pinecone, not PostgreSQL.

    Args:
        user_id: User's unique identifier
        doc_id: Unique document ID
        name: Document display name
        file_type: File type (pdf, docx, etc.)
        size_bytes: File size in bytes
        chunk_count: Number of chunks in Pinecone
        session_id: Session where document was uploaded
        pinecone_namespace: Pinecone namespace (defaults to user_id)
    """
    from sqlalchemy import select

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
                file_type=file_type,
                size_bytes=size_bytes,
                chunk_count=chunk_count,
                pinecone_namespace=pinecone_namespace or user_id
            )
            session.add(doc)
            await session.flush()  # Get the doc ID before creating link

        # Create document-session link
        doc_session = DocumentSession(
            document_id=doc_id,
            session_id=session_id
        )
        session.add(doc_session)

        await session.commit()
        return doc


async def get_user_documents(user_id: str, session_id: Optional[str] = None) -> List[dict]:
    """
    Get all documents for a user, optionally filtered by session.

    Args:
        user_id: User's unique identifier
        session_id: Optional session ID to filter documents
    """
    from sqlalchemy import select

    async with db.session() as session:
        if session_id:
            # Get documents for a specific session via junction table
            query = select(Document).join(DocumentSession).where(
                Document.user_id == user_id,
                DocumentSession.session_id == session_id
            )
        else:
            # Get all user documents
            query = select(Document).where(Document.user_id == user_id)

        result = await session.execute(query)
        docs = result.scalars().all()

        return [
            {
                "id": doc.id,
                "name": doc.name,
                "file_type": doc.file_type,
                "created_at": doc.created_at.isoformat(),
                "size_bytes": doc.size_bytes,
                "chunk_count": doc.chunk_count,
                "pinecone_namespace": doc.pinecone_namespace
            }
            for doc in docs
        ]


async def get_session_documents(session_id: str) -> List[dict]:
    """Get all documents linked to a specific session"""
    from sqlalchemy import select

    async with db.session() as session:
        query = select(Document).join(DocumentSession).where(
            DocumentSession.session_id == session_id
        )
        result = await session.execute(query)
        docs = result.scalars().all()

        return [
            {
                "id": doc.id,
                "name": doc.name,
                "file_type": doc.file_type,
                "created_at": doc.created_at.isoformat(),
                "size_bytes": doc.size_bytes,
                "chunk_count": doc.chunk_count
            }
            for doc in docs
        ]


async def link_document_to_session(document_id: str, session_id: str) -> bool:
    """Link an existing document to a session (for reusing documents across sessions)"""
    from sqlalchemy import select

    async with db.session() as session:
        # Check if link already exists
        result = await session.execute(
            select(DocumentSession).where(
                DocumentSession.document_id == document_id,
                DocumentSession.session_id == session_id
            )
        )
        if result.scalar_one_or_none():
            return True  # Already linked

        # Create new link
        doc_session = DocumentSession(
            document_id=document_id,
            session_id=session_id
        )
        session.add(doc_session)
        await session.commit()
        return True


async def delete_document_from_db(user_id: str, doc_id: str) -> bool:
    """Delete a document and its session links"""
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

        # Delete session links first (cascade should handle this, but explicit is safer)
        await session.execute(
            delete(DocumentSession).where(DocumentSession.document_id == doc_id)
        )

        # Delete document
        await session.execute(
            delete(Document).where(Document.id == doc_id)
        )

        await session.commit()
        return True


async def unlink_document_from_session(document_id: str, session_id: str) -> bool:
    """Remove a document link from a session (document remains in other sessions)"""
    from sqlalchemy import delete

    async with db.session() as session:
        await session.execute(
            delete(DocumentSession).where(
                DocumentSession.document_id == document_id,
                DocumentSession.session_id == session_id
            )
        )
        await session.commit()
        return True


# ============= Session Management Functions =============

async def create_session(user_id: str, title: Optional[str] = None) -> MeetingSession:
    """Create a new meeting session for a user"""
    import uuid

    session_id = str(uuid.uuid4())
    if not title:
        # Generate default title with date
        title = f"Interview {datetime.utcnow().strftime('%b %d, %Y %H:%M')}"

    async with db.session() as session:
        meeting_session = MeetingSession(
            id=session_id,
            user_id=user_id,
            title=title,
            status="active",
            transcript_entries=[],
            detected_questions=[],
            answers=[],
            document_ids=[],
            full_transcript=""
        )
        session.add(meeting_session)
        await session.commit()
        await session.refresh(meeting_session)
        return meeting_session


async def get_session(session_id: str, user_id: str) -> Optional[MeetingSession]:
    """Get a session by ID, verifying ownership"""
    from sqlalchemy import select

    async with db.session() as session:
        result = await session.execute(
            select(MeetingSession).where(
                MeetingSession.id == session_id,
                MeetingSession.user_id == user_id
            )
        )
        return result.scalar_one_or_none()


async def get_user_sessions(user_id: str, include_archived: bool = False) -> List[dict]:
    """Get all sessions for a user, ordered by most recent"""
    from sqlalchemy import select, desc

    async with db.session() as session:
        query = select(MeetingSession).where(
            MeetingSession.user_id == user_id
        )

        if not include_archived:
            query = query.where(MeetingSession.status == "active")

        query = query.order_by(desc(MeetingSession.updated_at))

        result = await session.execute(query)
        sessions = result.scalars().all()

        return [
            {
                "id": s.id,
                "title": s.title,
                "status": s.status,
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "updated_at": s.updated_at.isoformat() if s.updated_at else None,
                "question_count": len(s.detected_questions) if s.detected_questions else 0,
                "has_transcript": bool(s.full_transcript)
            }
            for s in sessions
        ]


async def update_session(
    session_id: str,
    user_id: str,
    title: Optional[str] = None,
    status: Optional[str] = None,
    transcript_entries: Optional[list] = None,
    detected_questions: Optional[list] = None,
    answers: Optional[list] = None,
    document_ids: Optional[list] = None,
    full_transcript: Optional[str] = None,
    conversation_summaries: Optional[list] = None,
    ended_at: Optional[datetime] = None
) -> Optional[MeetingSession]:
    """Update a session's data"""
    from sqlalchemy import select

    async with db.session() as session:
        result = await session.execute(
            select(MeetingSession).where(
                MeetingSession.id == session_id,
                MeetingSession.user_id == user_id
            )
        )
        meeting_session = result.scalar_one_or_none()

        if not meeting_session:
            return None

        if title is not None:
            meeting_session.title = title
        if status is not None:
            meeting_session.status = status
        if transcript_entries is not None:
            meeting_session.transcript_entries = transcript_entries
        if detected_questions is not None:
            meeting_session.detected_questions = detected_questions
        if answers is not None:
            meeting_session.answers = answers
        if document_ids is not None:
            meeting_session.document_ids = document_ids
        if full_transcript is not None:
            meeting_session.full_transcript = full_transcript
        if conversation_summaries is not None:
            meeting_session.conversation_summaries = conversation_summaries
        if ended_at is not None:
            meeting_session.ended_at = ended_at

        meeting_session.updated_at = datetime.utcnow()
        await session.commit()
        await session.refresh(meeting_session)
        return meeting_session


async def append_to_session(
    session_id: str,
    user_id: str,
    transcript_entry: Optional[dict] = None,
    question: Optional[dict] = None,
    answer: Optional[dict] = None,
    append_transcript_text: Optional[str] = None
) -> bool:
    """Append data to a session (for real-time updates)"""
    from sqlalchemy import select

    async with db.session() as session:
        result = await session.execute(
            select(MeetingSession).where(
                MeetingSession.id == session_id,
                MeetingSession.user_id == user_id
            )
        )
        meeting_session = result.scalar_one_or_none()

        if not meeting_session:
            return False

        if transcript_entry:
            entries = meeting_session.transcript_entries or []
            entries.append(transcript_entry)
            meeting_session.transcript_entries = entries

        if question:
            questions = meeting_session.detected_questions or []
            questions.append(question)
            meeting_session.detected_questions = questions

        if answer:
            answers = meeting_session.answers or []
            answers.append(answer)
            meeting_session.answers = answers

        if append_transcript_text:
            meeting_session.full_transcript = (meeting_session.full_transcript or "") + append_transcript_text

        meeting_session.updated_at = datetime.utcnow()
        await session.commit()
        return True


async def delete_session(session_id: str, user_id: str) -> bool:
    """Delete a session"""
    from sqlalchemy import select, delete as sql_delete

    async with db.session() as session:
        # Verify ownership
        result = await session.execute(
            select(MeetingSession).where(
                MeetingSession.id == session_id,
                MeetingSession.user_id == user_id
            )
        )
        meeting_session = result.scalar_one_or_none()

        if not meeting_session:
            return False

        await session.execute(
            sql_delete(MeetingSession).where(MeetingSession.id == session_id)
        )
        await session.commit()
        return True


async def archive_session(session_id: str, user_id: str) -> bool:
    """Archive a session instead of deleting"""
    result = await update_session(session_id, user_id, status="archived", ended_at=datetime.utcnow())
    return result is not None

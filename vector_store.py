"""
Vector Store - Pinecone Integration

This module provides vector storage and semantic search using Pinecone.
All document embeddings are stored in Pinecone for both development and production.

Each chunk includes comprehensive metadata for filtering:
- document_id, document_name, user_id
- file_type, file_size_bytes
- chunk_index, total_chunks
- created_at, updated_at
- Custom metadata fields

Required environment variables:
- PINECONE_API_KEY: Your Pinecone API key
- PINECONE_INDEX_NAME: Index name (default: "meeting-assistant")
- PINECONE_REGION: AWS region (default: "us-east-1")
"""

import os
import json
import hashlib
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

# Import database functions for persistence
from database import save_document_to_db, delete_document_from_db, log_usage, log_audit


class FileType(str, Enum):
    """Supported file types"""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "md"
    CSV = "csv"
    XLSX = "xlsx"
    IMAGE = "image"
    UNKNOWN = "unknown"


@dataclass
class DocumentMetadata:
    """Rich metadata for documents"""
    file_name: str
    file_type: str
    file_size_bytes: int
    file_hash: str  # MD5 hash of original content for deduplication
    total_chunks: int
    created_at: str  # ISO format
    updated_at: str  # ISO format
    source: str = "upload"  # upload, api, etc.
    tags: List[str] = field(default_factory=list)
    custom: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_name": self.file_name,
            "file_type": self.file_type,
            "file_size_bytes": self.file_size_bytes,
            "file_hash": self.file_hash,
            "total_chunks": self.total_chunks,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source": self.source,
            "tags": self.tags,
            **self.custom
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentMetadata":
        known_keys = {"file_name", "file_type", "file_size_bytes", "file_hash",
                      "total_chunks", "created_at", "updated_at", "source", "tags"}
        custom = {k: v for k, v in data.items() if k not in known_keys}
        return cls(
            file_name=data.get("file_name", ""),
            file_type=data.get("file_type", "unknown"),
            file_size_bytes=data.get("file_size_bytes", 0),
            file_hash=data.get("file_hash", ""),
            total_chunks=data.get("total_chunks", 0),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            source=data.get("source", "upload"),
            tags=data.get("tags", []),
            custom=custom
        )


@dataclass
class ChunkMetadata:
    """Rich metadata for individual chunks"""
    document_id: str
    document_name: str
    user_id: str
    chunk_index: int
    total_chunks: int
    chunk_size: int  # Character count
    file_type: str
    file_name: str
    created_at: str  # ISO format
    updated_at: str  # ISO format
    start_char: int  # Starting character position in original document
    end_char: int  # Ending character position
    tags: List[str] = field(default_factory=list)
    custom: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "document_name": self.document_name,
            "user_id": self.user_id,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "chunk_size": self.chunk_size,
            "file_type": self.file_type,
            "file_name": self.file_name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "tags": self.tags,
            **self.custom
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkMetadata":
        known_keys = {"document_id", "document_name", "user_id", "chunk_index",
                      "total_chunks", "chunk_size", "file_type", "file_name",
                      "created_at", "updated_at", "start_char", "end_char", "tags"}
        custom = {k: v for k, v in data.items() if k not in known_keys}
        return cls(
            document_id=data.get("document_id", ""),
            document_name=data.get("document_name", ""),
            user_id=data.get("user_id", ""),
            chunk_index=data.get("chunk_index", 0),
            total_chunks=data.get("total_chunks", 0),
            chunk_size=data.get("chunk_size", 0),
            file_type=data.get("file_type", "unknown"),
            file_name=data.get("file_name", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            start_char=data.get("start_char", 0),
            end_char=data.get("end_char", 0),
            tags=data.get("tags", []),
            custom=custom
        )


@dataclass
class VectorDocument:
    """Represents a document stored in the vector store"""
    id: str
    user_id: str
    name: str
    content: str
    metadata: DocumentMetadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    chunks: List[Dict[str, Any]] = field(default_factory=list)  # For tracking chunk data

    @property
    def uploaded_at(self) -> datetime:
        """Alias for created_at for compatibility"""
        return self.created_at


@dataclass
class VectorChunk:
    """Represents a chunk of a document with its embedding"""
    id: str
    document_id: str
    user_id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: ChunkMetadata = None


@dataclass
class SearchResult:
    """Search result from vector store"""
    chunk_id: str
    document_id: str
    document_name: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetadataFilter:
    """Filter for metadata-based search"""
    file_type: Optional[str] = None
    file_name: Optional[str] = None
    tags: Optional[List[str]] = None
    created_after: Optional[str] = None  # ISO format
    created_before: Optional[str] = None  # ISO format
    custom: Optional[Dict[str, Any]] = None

    def to_pinecone_filter(self) -> Dict[str, Any]:
        """Convert to Pinecone filter format"""
        filters = {}

        if self.file_type:
            filters["file_type"] = {"$eq": self.file_type}
        if self.file_name:
            filters["file_name"] = {"$eq": self.file_name}
        if self.tags:
            filters["tags"] = {"$in": self.tags}
        if self.created_after:
            filters["created_at"] = {"$gte": self.created_after}
        if self.created_before:
            if "created_at" in filters:
                filters["created_at"]["$lte"] = self.created_before
            else:
                filters["created_at"] = {"$lte": self.created_before}
        if self.custom:
            for key, value in self.custom.items():
                filters[key] = {"$eq": value}

        return filters if filters else None


class PineconeVectorStore:
    """
    Pinecone vector store for document embeddings.

    All documents are stored in Pinecone with user-specific namespaces.
    Each user's documents are isolated in their own namespace.

    Usage:
        store = PineconeVectorStore()

        # Add document with metadata
        doc = await store.add_document(
            user_id="user123",
            name="Report Q4",
            content="...",
            file_name="report_q4.pdf",
            file_type="pdf",
            tags=["finance", "quarterly"]
        )

        # Search with metadata filter
        results = await store.search(
            user_id="user123",
            query="quarterly revenue",
            filter=MetadataFilter(file_type="pdf", tags=["finance"])
        )

        # Delete by file name
        await store.delete_by_file_name(user_id="user123", file_name="report_q4.pdf")
    """

    def __init__(self):
        self._pc = None
        self._index = None
        self._index_name = os.getenv("PINECONE_INDEX_NAME", "meeting-assistant")
        self._dimension = 1536  # OpenAI text-embedding-3-small dimension
        self._openai: Optional[OpenAI] = None
        # Local cache for document metadata (Pinecone only stores chunk metadata)
        self._doc_metadata_cache: Dict[str, Dict[str, VectorDocument]] = {}

    def _get_pinecone_client(self):
        """Get or create Pinecone client"""
        if self._pc is None:
            from pinecone import Pinecone
            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                raise ValueError(
                    "PINECONE_API_KEY not set. Please add your Pinecone API key to .env file.\n"
                    "Get your key from: https://app.pinecone.io"
                )
            self._pc = Pinecone(api_key=api_key)
        return self._pc

    def _get_index(self):
        """Get or create Pinecone index"""
        if self._index is None:
            pc = self._get_pinecone_client()

            # Check if index exists, create if not
            existing_indexes = [idx.name for idx in pc.list_indexes()]

            if self._index_name not in existing_indexes:
                from pinecone import ServerlessSpec
                pc.create_index(
                    name=self._index_name,
                    dimension=self._dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=os.getenv("PINECONE_REGION", "us-east-1")
                    )
                )
                print(f"Created Pinecone index: {self._index_name}")

            self._index = pc.Index(self._index_name)
        return self._index

    def _get_openai(self) -> OpenAI:
        """Get or create OpenAI client"""
        if self._openai is None:
            self._openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._openai

    def _get_file_type(self, file_name: str) -> str:
        """Detect file type from extension"""
        ext = file_name.lower().split('.')[-1] if '.' in file_name else ""
        type_map = {
            "pdf": FileType.PDF.value,
            "docx": FileType.DOCX.value,
            "doc": FileType.DOCX.value,
            "txt": FileType.TXT.value,
            "md": FileType.MD.value,
            "csv": FileType.CSV.value,
            "xlsx": FileType.XLSX.value,
            "xls": FileType.XLSX.value,
            "png": FileType.IMAGE.value,
            "jpg": FileType.IMAGE.value,
            "jpeg": FileType.IMAGE.value,
            "gif": FileType.IMAGE.value,
            "webp": FileType.IMAGE.value,
        }
        return type_map.get(ext, FileType.UNKNOWN.value)

    def _compute_hash(self, content: str) -> str:
        """Compute MD5 hash of content for deduplication"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def _chunk_text(
        self,
        text: str,
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks with position tracking"""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append({
                "content": text[start:end],
                "start_char": start,
                "end_char": end
            })
            start = end - overlap
            if start + overlap >= len(text):
                break
        return chunks

    def is_configured(self) -> bool:
        """Check if Pinecone is properly configured"""
        return bool(os.getenv("PINECONE_API_KEY"))

    async def add_document(
        self,
        user_id: str,
        name: str,
        content: str,
        session_id: str,
        file_name: Optional[str] = None,
        file_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        source: str = "upload",
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> VectorDocument:
        """
        Add a document to Pinecone with comprehensive metadata.

        Args:
            user_id: User's unique identifier (used as Pinecone namespace)
            name: Display name for the document
            content: Full text content of the document
            session_id: Session ID where document is being uploaded (required)
            file_name: Original file name (used for deletion by name)
            file_type: File type (pdf, docx, etc.) - auto-detected if not provided
            tags: List of tags for filtering
            source: Source of the document (upload, api, etc.)
            custom_metadata: Additional custom metadata fields

        Returns:
            VectorDocument with full metadata
        """
        if not self.is_configured():
            raise ValueError(
                "Pinecone is not configured. Please set PINECONE_API_KEY in your .env file."
            )

        now = datetime.utcnow()
        now_iso = now.isoformat()
        doc_id = f"doc_{int(now.timestamp())}_{uuid.uuid4().hex[:8]}"

        # Use file_name or fall back to name
        actual_file_name = file_name or name

        # Auto-detect file type if not provided
        actual_file_type = file_type or self._get_file_type(actual_file_name)

        # Chunk the content
        text_chunks = self._chunk_text(content)
        total_chunks = len(text_chunks)

        # Build document metadata
        doc_metadata = DocumentMetadata(
            file_name=actual_file_name,
            file_type=actual_file_type,
            file_size_bytes=len(content.encode('utf-8')),
            file_hash=self._compute_hash(content),
            total_chunks=total_chunks,
            created_at=now_iso,
            updated_at=now_iso,
            source=source,
            tags=tags or [],
            custom=custom_metadata or {}
        )

        document = VectorDocument(
            id=doc_id,
            user_id=user_id,
            name=name,
            content=content,
            metadata=doc_metadata,
            created_at=now,
            updated_at=now
        )

        # Cache document metadata locally
        if user_id not in self._doc_metadata_cache:
            self._doc_metadata_cache[user_id] = {}
        self._doc_metadata_cache[user_id][doc_id] = document

        # Create chunks with embeddings
        index = self._get_index()
        vectors = []

        for i, chunk_data in enumerate(text_chunks):
            chunk_id = f"{doc_id}_chunk_{i}"

            # Generate embedding
            try:
                response = self._get_openai().embeddings.create(
                    model="text-embedding-3-small",
                    input=chunk_data["content"]
                )
                embedding = response.data[0].embedding
            except Exception as e:
                print(f"Embedding error: {e}")
                continue

            # Build chunk metadata
            chunk_metadata = ChunkMetadata(
                document_id=doc_id,
                document_name=name,
                user_id=user_id,
                chunk_index=i,
                total_chunks=total_chunks,
                chunk_size=len(chunk_data["content"]),
                file_type=actual_file_type,
                file_name=actual_file_name,
                created_at=now_iso,
                updated_at=now_iso,
                start_char=chunk_data["start_char"],
                end_char=chunk_data["end_char"],
                tags=tags or [],
                custom=custom_metadata or {}
            )

            # Prepare vector for Pinecone
            metadata_dict = chunk_metadata.to_dict()
            # Truncate content to fit Pinecone metadata limits (40KB)
            metadata_dict["content"] = chunk_data["content"][:3500]

            vectors.append({
                "id": chunk_id,
                "values": embedding,
                "metadata": metadata_dict
            })

        # Upsert vectors in batches
        if vectors:
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                index.upsert(vectors=batch, namespace=user_id)

        # Store chunks in document for reference
        document.chunks = text_chunks

        # Save to PostgreSQL database for persistence (metadata only, no content)
        try:
            await save_document_to_db(
                user_id=user_id,
                doc_id=doc_id,
                name=name,
                file_type=actual_file_type,
                size_bytes=len(content.encode('utf-8')),
                chunk_count=total_chunks,
                session_id=session_id,
                pinecone_namespace=user_id
            )
            print(f"✓ Document {doc_id} saved to PostgreSQL (metadata only)")

            # Log usage for embeddings
            await log_usage(
                user_id=user_id,
                service="openai",
                operation="embedding",
                tokens=len(content.split()) * len(text_chunks),  # Approximate token count
                session_id=session_id
            )

            # Audit log for document upload
            await log_audit(
                action="document_upload",
                user_id=user_id,
                resource_type="document",
                resource_id=doc_id,
                extra_data={
                    "name": name,
                    "file_type": actual_file_type,
                    "size_bytes": len(content.encode('utf-8')),
                    "chunks": total_chunks,
                    "session_id": session_id
                }
            )
        except Exception as e:
            print(f"Warning: Failed to save document to PostgreSQL: {e}")

        return document

    async def search(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        filter: Optional[MetadataFilter] = None
    ) -> List[SearchResult]:
        """
        Search for similar documents with optional metadata filtering.

        Args:
            user_id: User's unique identifier (Pinecone namespace)
            query: Search query text
            top_k: Maximum number of results to return
            filter: Optional metadata filter

        Returns:
            List of SearchResult objects
        """
        if not self.is_configured():
            return []

        try:
            # Generate query embedding
            response = self._get_openai().embeddings.create(
                model="text-embedding-3-small",
                input=query
            )
            query_embedding = response.data[0].embedding
        except Exception as e:
            print(f"Query embedding error: {e}")
            return []

        try:
            index = self._get_index()

            query_params = {
                "vector": query_embedding,
                "top_k": top_k,
                "include_metadata": True,
                "namespace": user_id
            }

            # Add metadata filter if provided
            if filter:
                pinecone_filter = filter.to_pinecone_filter()
                if pinecone_filter:
                    query_params["filter"] = pinecone_filter

            results = index.query(**query_params)

            search_results = []
            for match in results.get("matches", []):
                metadata = match.get("metadata", {})
                search_results.append(SearchResult(
                    chunk_id=match["id"],
                    document_id=metadata.get("document_id", ""),
                    document_name=metadata.get("document_name", ""),
                    content=metadata.get("content", ""),
                    score=match["score"],
                    metadata=metadata
                ))

            return search_results
        except Exception as e:
            print(f"Pinecone search error: {e}")
            return []

    async def search_by_file_type(
        self,
        user_id: str,
        query: str,
        file_type: str,
        top_k: int = 5
    ) -> List[SearchResult]:
        """Search within documents of a specific file type"""
        filter = MetadataFilter(file_type=file_type)
        return await self.search(user_id, query, top_k, filter)

    async def search_by_tags(
        self,
        user_id: str,
        query: str,
        tags: List[str],
        top_k: int = 5
    ) -> List[SearchResult]:
        """Search within documents matching any of the specified tags"""
        filter = MetadataFilter(tags=tags)
        return await self.search(user_id, query, top_k, filter)

    async def delete_document(self, user_id: str, document_id: str) -> bool:
        """Delete a document and all its chunks by document ID"""
        if not self.is_configured():
            return False

        try:
            index = self._get_index()

            # Delete by document_id filter
            index.delete(
                filter={"document_id": {"$eq": document_id}},
                namespace=user_id
            )

            # Remove from local cache
            if user_id in self._doc_metadata_cache:
                self._doc_metadata_cache[user_id].pop(document_id, None)

            # Delete from PostgreSQL database
            try:
                await delete_document_from_db(user_id, document_id)
                print(f"✓ Document {document_id} deleted from PostgreSQL")

                # Audit log for document deletion
                await log_audit(
                    action="document_delete",
                    user_id=user_id,
                    resource_type="document",
                    resource_id=document_id
                )
            except Exception as db_err:
                print(f"Warning: Failed to delete document from PostgreSQL: {db_err}")

            return True
        except Exception as e:
            print(f"Pinecone delete error: {e}")
            return False

    async def delete_by_file_name(self, user_id: str, file_name: str) -> bool:
        """Delete all chunks for documents with a specific file name"""
        if not self.is_configured():
            return False

        try:
            index = self._get_index()
            index.delete(
                filter={"file_name": {"$eq": file_name}},
                namespace=user_id
            )
            return True
        except Exception as e:
            print(f"Pinecone delete by file_name error: {e}")
            return False

    async def delete_by_file_type(self, user_id: str, file_type: str) -> bool:
        """Delete all documents of a specific file type"""
        if not self.is_configured():
            return False

        try:
            index = self._get_index()
            index.delete(
                filter={"file_type": {"$eq": file_type}},
                namespace=user_id
            )
            return True
        except Exception as e:
            print(f"Pinecone delete by file_type error: {e}")
            return False

    async def delete_by_tags(self, user_id: str, tags: List[str]) -> bool:
        """Delete all documents matching any of the specified tags"""
        if not self.is_configured():
            return False

        try:
            index = self._get_index()
            index.delete(
                filter={"tags": {"$in": tags}},
                namespace=user_id
            )
            return True
        except Exception as e:
            print(f"Pinecone delete by tags error: {e}")
            return False

    async def list_documents(self, user_id: str) -> List[Dict[str, Any]]:
        """List all documents for a user with full metadata"""
        if not self.is_configured():
            return []

        try:
            index = self._get_index()

            # Get index stats for this namespace
            stats = index.describe_index_stats()
            namespace_stats = stats.get("namespaces", {}).get(user_id, {})

            if namespace_stats.get("vector_count", 0) == 0:
                return []

            # Query with a zero vector to get all documents
            zero_vector = [0.0] * self._dimension
            results = index.query(
                vector=zero_vector,
                top_k=10000,
                include_metadata=True,
                namespace=user_id
            )

            # Deduplicate by document_id
            docs = {}
            for match in results.get("matches", []):
                metadata = match.get("metadata", {})
                doc_id = metadata.get("document_id")
                if doc_id and doc_id not in docs:
                    docs[doc_id] = {
                        "id": doc_id,
                        "name": metadata.get("document_name", ""),
                        "file_name": metadata.get("file_name", ""),
                        "file_type": metadata.get("file_type", "unknown"),
                        "total_chunks": metadata.get("total_chunks", 0),
                        "created_at": metadata.get("created_at", ""),
                        "updated_at": metadata.get("updated_at", ""),
                        "tags": metadata.get("tags", []),
                    }

            return list(docs.values())
        except Exception as e:
            print(f"Pinecone list error: {e}")
            return []

    async def get_document(self, user_id: str, document_id: str) -> Optional[VectorDocument]:
        """Get a document by ID from cache"""
        if user_id in self._doc_metadata_cache:
            return self._doc_metadata_cache[user_id].get(document_id)
        return None

    async def get_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics about user's documents"""
        if not self.is_configured():
            return {"total_vectors": 0, "total_documents": 0}

        try:
            index = self._get_index()
            stats = index.describe_index_stats()
            namespace_stats = stats.get("namespaces", {}).get(user_id, {})

            docs = await self.list_documents(user_id)

            # Count by file type
            type_counts = {}
            for doc in docs:
                file_type = doc.get("file_type", "unknown")
                type_counts[file_type] = type_counts.get(file_type, 0) + 1

            return {
                "total_vectors": namespace_stats.get("vector_count", 0),
                "total_documents": len(docs),
                "documents_by_type": type_counts,
            }
        except Exception as e:
            print(f"Pinecone stats error: {e}")
            return {"total_vectors": 0, "total_documents": 0}

    def get_context(self, search_results: List[SearchResult]) -> str:
        """Format search results as context for LLM"""
        if not search_results:
            return ""

        parts = []
        for r in search_results:
            header = f"[From: {r.document_name}"
            if r.metadata.get("file_type"):
                header += f" ({r.metadata['file_type']})"
            header += "]"
            parts.append(f"{header}\n{r.content}")

        return "\n\n---\n\n".join(parts)


# Global instance
vector_store = PineconeVectorStore()

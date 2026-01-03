import os
import json
import asyncio
import logging
import io
import csv
import base64
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException, Depends, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv

from deepgram import Deepgram
from openai import OpenAI
from pypdf import PdfReader
from docx import Document as DocxDocument
import openpyxl
import anthropic

from vector_store import PineconeVectorStore
from question_detector import detect_question
from auth import clerk_auth, ClerkUser, require_auth, optional_auth
from database import db, get_or_create_user


# User-specific vector stores
_user_vector_stores: dict = {}


def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """Extract text from various file formats"""
    extension = filename.lower().split('.')[-1]

    try:
        # PDF files
        if extension == 'pdf':
            pdf_reader = PdfReader(io.BytesIO(file_bytes))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()

        # DOCX files
        elif extension == 'docx':
            doc = DocxDocument(io.BytesIO(file_bytes))
            text = "\n".join([para.text for para in doc.paragraphs])
            return text.strip()

        # Markdown files
        elif extension == 'md':
            return file_bytes.decode('utf-8')

        # CSV files
        elif extension == 'csv':
            text = file_bytes.decode('utf-8')
            reader = csv.reader(io.StringIO(text))
            rows = [', '.join(row) for row in reader]
            return '\n'.join(rows)

        # Excel files
        elif extension in ['xlsx', 'xls']:
            wb = openpyxl.load_workbook(io.BytesIO(file_bytes))
            text = ""
            for sheet_name in wb.sheetnames:
                sheet = wb[sheet_name]
                text += f"\n=== Sheet: {sheet_name} ===\n"
                for row in sheet.iter_rows(values_only=True):
                    text += ', '.join([str(cell) if cell is not None else '' for cell in row]) + '\n'
            return text.strip()

        # Image files - use Claude Vision
        elif extension in ['png', 'jpg', 'jpeg', 'gif', 'webp']:
            return analyze_image_with_claude(file_bytes, extension)

        # Plain text files (txt, py, js, json, etc.)
        else:
            return file_bytes.decode('utf-8')

    except Exception as e:
        raise ValueError(f"Error parsing {extension} file: {str(e)}")


def analyze_image_with_claude(image_bytes: bytes, extension: str) -> str:
    """Analyze image using OpenAI gpt-4o-mini Vision API"""
    try:
        # Convert image to base64
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')

        # Determine media type for data URL
        media_type_map = {
            'png': 'image/png',
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'gif': 'image/gif',
            'webp': 'image/webp'
        }
        media_type = media_type_map.get(extension, 'image/jpeg')

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=2000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "Analyze this image in detail. Describe what you see, any text present, diagrams, charts, or other relevant information. Provide a comprehensive description that can be used for semantic search."
                        }
                    ]
                }
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"[Image file - analysis failed: {str(e)}]"


def get_vector_store_for_user(user_id: str) -> PineconeVectorStore:
    """Get the Pinecone vector store for a user"""
    if user_id not in _user_vector_stores:
        _user_vector_stores[user_id] = PineconeVectorStore()
    return _user_vector_stores[user_id]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(override=True)

# Global state
class MeetingState:
    def __init__(self):
        self.transcript: list[dict] = []  # [{speaker: int, text: str, timestamp: str}]
        self.full_transcript: str = ""
        self.detected_questions: list[dict] = []
        self.suggestions: list[dict] = []
        self.last_analysis_time: float = 0
        self.connected_clients: set[WebSocket] = set()
        self.last_processed_transcript_length: int = 0  # Track processed length
    
    def add_utterance(self, speaker: int, text: str):
        entry = {
            "speaker": speaker,
            "text": text,
            "timestamp": datetime.now().isoformat()
        }
        self.transcript.append(entry)
        self.full_transcript += f" {text}"
    
    def get_recent_transcript(self, chars: int = 3000) -> str:
        return self.full_transcript[-chars:] if len(self.full_transcript) > chars else self.full_transcript

    def get_new_transcript_for_detection(self, max_chars: int = 2500) -> tuple[str, bool]:
        """Get only new transcript content that hasn't been processed yet.
        Returns (transcript_text, has_new_content)"""
        current_length = len(self.full_transcript)

        # No new content
        if current_length <= self.last_processed_transcript_length:
            return ("", False)

        # Get new portion with some context (last 1000 chars before new content for overlap)
        context_start = max(0, self.last_processed_transcript_length - 1000)
        new_text = self.full_transcript[context_start:]

        # Update processed length
        self.last_processed_transcript_length = current_length

        # Limit to max_chars
        result = new_text[-max_chars:] if len(new_text) > max_chars else new_text
        return (result, True)

    def clear(self):
        self.transcript = []
        self.full_transcript = ""
        self.detected_questions = []
        self.suggestions = []
        self.last_processed_transcript_length = 0

meeting_state = MeetingState()

# Global periodic tasks
async def global_periodic_question_detection():
    """Check for questions every 4 seconds using Haiku"""
    logger.info("✓ Global periodic question detection started")
    while True:
        await asyncio.sleep(4)
        await detect_questions_with_haiku()

async def global_periodic_suggestions():
    """Generate suggestions every 15 seconds"""
    logger.info("✓ Global periodic suggestions started")
    while True:
        await asyncio.sleep(5)
        await analyze_transcript()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database
    logger.info("Initializing database...")
    await db.init_db()
    logger.info("✓ Database initialized")

    # Start periodic tasks when app starts
    logger.info("Starting global periodic tasks...")
    question_task = asyncio.create_task(global_periodic_question_detection())
    # suggestion_task = asyncio.create_task(global_periodic_suggestions())  # Disabled - only detect questions
    logger.info("✓ Global periodic tasks running")
    yield
    # Cancel tasks on shutdown
    question_task.cancel()
    # suggestion_task.cancel()  # Disabled
    # Close database
    await db.close()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication endpoints
@app.get("/api/auth/config")
async def get_auth_config():
    """Return Clerk configuration for frontend"""
    publishable_key = os.getenv("CLERK_PUBLISHABLE_KEY", "")
    return {
        "publishableKey": publishable_key,
        "isConfigured": clerk_auth.is_configured()
    }

@app.get("/api/auth/me")
async def get_current_user_info(request: Request):
    """Get current authenticated user info"""
    if not clerk_auth.is_configured():
        # Return a mock user when Clerk is not configured (dev mode)
        return {
            "authenticated": True,
            "user": {
                "id": "dev_user",
                "email": "dev@localhost",
                "fullName": "Development User"
            },
            "devMode": True
        }

    user = await optional_auth(request)
    if user:
        return {
            "authenticated": True,
            "user": {
                "id": user.user_id,
                "email": user.email,
                "fullName": user.full_name,
                "firstName": user.first_name,
                "lastName": user.last_name,
                "imageUrl": user.image_url
            },
            "devMode": False
        }
    return {"authenticated": False, "user": None, "devMode": False}


async def get_user_id_from_request(request: Request) -> str:
    """Get user ID from request, returns 'dev_user' if auth not configured"""
    if not clerk_auth.is_configured():
        return "dev_user"

    user = await optional_auth(request)
    if user:
        # Persist user to database
        await get_or_create_user(
            user_id=user.user_id,
            email=user.email,
            first_name=user.first_name,
            last_name=user.last_name,
            image_url=user.image_url
        )
        return user.user_id
    return "anonymous"


async def get_user_id_from_websocket(websocket: WebSocket, token: str = None) -> str:
    """Get user ID from WebSocket, returns 'dev_user' if auth not configured"""
    if not clerk_auth.is_configured():
        return "dev_user"

    user = await clerk_auth.verify_websocket(websocket, token)
    if user:
        # Persist user to database
        await get_or_create_user(
            user_id=user.user_id,
            email=user.email,
            first_name=user.first_name,
            last_name=user.last_name,
            image_url=user.image_url
        )
        return user.user_id
    return "anonymous"

# Broadcast to all connected frontend clients
async def broadcast(message: dict):
    disconnected = set()
    for client in meeting_state.connected_clients:
        try:
            await client.send_json(message)
        except:
            disconnected.add(client)
    meeting_state.connected_clients -= disconnected

# Force question detection on selected transcript
async def force_detect_questions(text: str):
    """Force question detection on user-selected transcript text - generates questions FROM the text"""
    if len(text) < 10:
        return

    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        system = """You are a question detector AND generator for interview preparation. Given a transcript excerpt:

YOUR TASK:
1. DETECT any questions already being asked in the text (direct or implicit)
2. GENERATE additional relevant interview questions based on topics mentioned

DETECTION RULES:
- Direct questions: "What is Docker?", "How does it work?"
- Implicit questions: "Tell me about...", "Explain...", "I want to know about..." → convert to proper questions

GENERATION RULES:
- Generate 1-2 interview questions about the main topics/technologies/projects mentioned
- Questions should be interview-style (e.g., "Can you explain...", "How did you implement...")

CRITICAL FORMATTING:
1. ALL questions MUST end with "?" - this is mandatory
2. Make questions specific to the content in the transcript
3. Combine fragmented questions into complete sentences

EXAMPLES:
- "tell me about your RIA project" → "Can you tell me about your RIA project?"
- Text mentions "Docker and Kubernetes" → Generate: "How did you use Docker and Kubernetes in your project?"
- "I want to understand the data pipeline" → "Can you explain your data pipeline architecture?"

Respond with a JSON object containing an array of questions:
{"questions": ["question 1?", "question 2?"]}"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=500,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Selected transcript:\n{text}"}
            ]
        )

        result_text = response.choices[0].message.content.strip()

        # Remove markdown code blocks if present
        if result_text.startswith('```'):
            lines = result_text.split('\n')
            result_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else result_text
            result_text = result_text.replace('```json', '').replace('```', '').strip()

        json_text = result_text.split('\n\n')[0].strip()
        result = json.loads(json_text)
        questions = result.get('questions', [])

        # Add all generated questions
        for question_text in questions:
            if question_text and question_text.strip().endswith('?'):
                # Check if question already exists
                exists = any(
                    existing['text'].lower() == question_text.lower()
                    for existing in meeting_state.detected_questions
                )
                if not exists:
                    q_entry = {
                        "text": question_text,
                        "speaker": 0,
                        "timestamp": datetime.now().isoformat(),
                        "source": "manual"
                    }
                    meeting_state.detected_questions.insert(0, q_entry)

                    logger.info(f"✓ Generated question from selection: {question_text}")
                    # Broadcast immediately
                    await broadcast({
                        "type": "question_detected",
                        "data": q_entry
                    })

    except Exception as e:
        logger.error(f"Force question detection error: {e}")

# Detect questions using GPT (smarter detection)
async def detect_questions_with_haiku():
    transcript, has_new = meeting_state.get_new_transcript_for_detection(2500)

    # Skip if no new content or too short
    if not has_new:
        logger.debug(f"Skipping question detection: no new transcript content")
        return

    if len(transcript) < 50:
        logger.debug(f"Skipping question detection: transcript too short ({len(transcript)} chars)")
        return

    logger.info(f"Running question detection on transcript ({len(transcript)} chars): {transcript[:100]}...")

    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        system = """You are a question detector. Analyze the transcript and identify ALL questions being asked by speakers, including both direct questions and implicit questions (requests for information).

TYPES OF QUESTIONS TO DETECT:
1. Direct questions: "What is Docker?", "How does it work?", "Can you explain?"
2. Implicit questions (requests for information): "I would like to know about...", "Tell me about...", "Explain..."

CRITICAL FORMATTING RULES:
1. ALL questions MUST end with "?" - this is mandatory
2. Convert implicit questions to proper question format
3. Questions must be at least 10 characters long
4. Combine fragmented questions across multiple utterances into ONE complete, grammatically correct question
5. IMPORTANT: Detect ALL questions in the transcript, including follow-up questions

EXAMPLES:
Input: "I would like to know more about the Docker implementation"
Output: "What would you like to know about the Docker implementation?" ✓

Input: "Tell me about Kubernetes"
Output: "Can you tell me about Kubernetes?" ✓

Input: "I want to understand how you implemented DevOps"
Output: "How did you implement DevOps?" ✓

Input: "Explain the architecture"
Output: "Can you explain the architecture?" ✓

Input: "What is the status of the project?"
Output: "What is the status of the project?" ✓

Input: "How did you implement the RIA and data pipeline?"
Output: "How did you implement the RIA and data pipeline?" ✓

COMBINING FRAGMENTED QUESTIONS:
- Split: "So I would like to know more about" + "the Docker and Kubernetes implementation"
  → Combined: "What would you like to know about the Docker and Kubernetes implementation?" ✓

INVALID formats to IGNORE (will NOT be detected):
✗ General statements without information request
✗ Comments, greetings, acknowledgments

Respond with a JSON object containing ALL questions found:
{"questions": ["question 1?", "question 2?", "question 3?"]}

If no questions found, return {"questions": []}"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=500,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Recent transcript:\n{transcript}"}
            ]
        )

        text = response.choices[0].message.content
        if not text:
            logger.warning("GPT returned empty response")
            return

        text = text.strip()
        logger.info(f"Raw GPT response: {text}")

        # Remove markdown code blocks if present
        if text.startswith('```'):
            # Extract JSON from code block
            lines = text.split('\n')
            text = '\n'.join(lines[1:-1]) if len(lines) > 2 else text
            text = text.replace('```json', '').replace('```', '').strip()

        # Extract only the JSON part (before any explanatory text after double newline)
        # Haiku sometimes adds explanations after the JSON
        json_text = text.split('\n\n')[0].strip()

        if not json_text:
            logger.warning("No JSON text found in GPT response")
            return

        result = json.loads(json_text)
        questions = result.get('questions', [])

        # Process all detected questions
        for question_text in questions:
            if question_text:
                # Validate that it's actually a question using our regex
                if not detect_question(question_text):
                    logger.info(f"Haiku detected non-question format, skipping: {question_text}")
                else:
                    # Check if question already exists
                    exists = any(
                        existing['text'].lower() == question_text.lower()
                        for existing in meeting_state.detected_questions
                    )
                    if not exists:
                        q_entry = {
                            "text": question_text,
                            "speaker": 0,
                            "timestamp": datetime.now().isoformat(),
                            "source": "haiku"
                        }
                        meeting_state.detected_questions.insert(0, q_entry)
                        meeting_state.detected_questions = meeting_state.detected_questions[:5]

                        logger.info(f"✓ Haiku detected question: {question_text}")
                        # Broadcast immediately
                        await broadcast({
                            "type": "question_detected",
                            "data": q_entry
                        })
    except Exception as e:
        logger.error(f"Haiku question detection error: {e}")
        if 'text' in locals():
            logger.error(f"Failed to parse Haiku response: {text[:200]}")

# Analyze transcript with Claude (generate suggestions)
# TODO: Phase 3 - This will be refactored to use user-specific document stores
# when meeting sessions become user-scoped
async def analyze_transcript():
    transcript = meeting_state.get_recent_transcript()
    if len(transcript) < 100:
        return

    try:
        # Generate suggestions
        # Note: Using dev_user store for now; will be user-scoped when meeting sessions are per-user
        store = get_vector_store_for_user("dev_user")
        search_results = await store.search("dev_user", transcript, top_k=3)
        doc_context = store.get_context(search_results)

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        doc_section = f"Relevant documents:\n{doc_context}" if doc_context else ""

        system = f"""You are an intelligent meeting assistant. Analyze the conversation transcript and suggest 2-4 highly relevant questions or topics.

{doc_section}

Respond with a JSON array only:
[{{"question": "...", "type": "clarification|context|insight|action"}}]"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=500,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Transcript:\n{transcript}"}
            ]
        )

        text = response.choices[0].message.content
        suggestions = json.loads(text)
        meeting_state.suggestions = suggestions
        await broadcast({"type": "suggestions", "data": suggestions})
    except Exception as e:
        logger.error(f"Analysis error: {e}")

# Answer question with GPT (streaming with proper event handling)
async def answer_question_stream(question: str, websocket: WebSocket, user_id: str = "anonymous"):
    full_answer = ""
    try:
        # Use user's vector store with more results for better context
        store = get_vector_store_for_user(user_id)
        logger.info(f"Searching Pinecone for user '{user_id}' with query: {question[:50]}...")
        search_results = await store.search(user_id, question, top_k=8)

        # Filter results by score threshold for better relevance
        relevant_results = [r for r in search_results if r.score > 0.3]
        logger.info(f"Found {len(search_results)} results, {len(relevant_results)} above threshold")

        # Log the results for debugging
        for i, r in enumerate(relevant_results):
            logger.info(f"  Result {i+1}: doc='{r.document_name}', score={r.score:.3f}, content={r.content[:100]}...")

        doc_context = store.get_context(relevant_results)

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        doc_section = f"""PROJECT DOCUMENTS (Use these to answer questions about the user's projects):
{doc_context}""" if doc_context else "No project documents uploaded yet."

        system = f"""You are an expert interview coach helping an Indian IT professional answer technical interview questions in real-time. Your responses should sound natural and conversational, like how an experienced Indian software engineer would explain things.

SPEAKING STYLE:
- Use natural, conversational English as spoken by Indian IT professionals
- Phrases like "So basically...", "What we did is...", "The thing is...", "Actually..." are fine
- Be confident and direct, like explaining to a colleague
- Use "we" when talking about project work (e.g., "We implemented...", "What we used is...")
- Avoid overly formal or robotic language

RESPONSE GUIDELINES:
1. **Start directly with the answer** - no preamble like "Great question" or "Sure"
2. **Be comprehensive but concise** - provide enough detail to demonstrate expertise (4-6 sentences for simple topics, more for complex ones)
3. **Use the document context** - when answering about projects, USE SPECIFIC DETAILS from the documents provided
4. **Structure for clarity** - use bullet points for multiple components
5. **Include concrete examples** - relate to the actual project when documents are available

WHEN DOCUMENTS ARE PROVIDED:
- Extract and cite SPECIFIC technical details (technologies, architectures, frameworks)
- Mention actual features, components, or implementations from the documents
- Reference specific aspects like data pipelines, APIs, databases, or workflows mentioned
- Provide depth by explaining HOW things work based on document details

RESPONSE FORMAT:
- For project questions: Start with a brief overview, then explain key technical components with specifics from documents
- For concept questions: Define clearly, explain how it works, give a practical example
- Use bullet points for listing multiple features or components

{doc_section}"""

        transcript = meeting_state.get_recent_transcript(2000)

        # Send start event
        await websocket.send_json({
            "type": "answer_start",
            "data": {"question": question}
        })

        # Stream the response with proper event handling
        logger.info(f"Starting stream for question: {question}")
        chunk_count = 0

        # Create stream and iterate chunks (OpenAI streaming)
        stream = client.chat.completions.create(
            model="gpt-4.1-mini",
            max_tokens=2000,
            stream=True,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Conversation context:\n{transcript}\n\nQuestion to answer: {question}\n\nProvide a detailed, expert-level response using any relevant document context."}
            ]
        )

        finish_reason = None
        for chunk in stream:
            if chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                chunk_count += 1
                full_answer += text
                logger.debug(f"Chunk {chunk_count}: {len(text)} chars")

                # Send chunk to client
                try:
                    await websocket.send_json({
                        "type": "answer_chunk",
                        "data": {
                            "question": question,
                            "chunk": text,
                            "is_final": False
                        }
                    })
                    # Small delay to ensure chunks are sent separately
                    await asyncio.sleep(0.01)
                except Exception as e:
                    logger.error(f"Error sending chunk: {e}")

            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

        logger.info(f"Stream complete. Total chunks: {chunk_count}")

        # Send final message with complete answer
        await websocket.send_json({
            "type": "answer_chunk",
            "data": {
                "question": question,
                "chunk": "",
                "is_final": True,
                "full_answer": full_answer,
                "stop_reason": finish_reason
            }
        })

    except Exception as e:
        logger.error(f"Answer error: {e}")
        import traceback
        traceback.print_exc()

        # Send error with partial answer if available
        error_msg = f"Error occurred: {str(e)}"
        if full_answer:
            error_msg = f"{full_answer}\n\n[Error: Response interrupted - {str(e)}]"

        await websocket.send_json({
            "type": "answer_chunk",
            "data": {
                "question": question,
                "chunk": "",
                "is_final": True,
                "full_answer": error_msg,
                "error": str(e)
            }
        })

# WebSocket for frontend clients (receive updates)
@app.websocket("/ws/client")
async def client_websocket(websocket: WebSocket, token: str = Query(None)):
    await websocket.accept()

    # Get user ID for this connection
    user_id = await get_user_id_from_websocket(websocket, token)
    logger.info(f"Client WebSocket connected: user_id={user_id}")

    meeting_state.connected_clients.add(websocket)

    # Send current state along with user info
    await websocket.send_json({
        "type": "init",
        "data": {
            "transcript": meeting_state.transcript,
            "questions": meeting_state.detected_questions,
            "suggestions": meeting_state.suggestions,
            "userId": user_id
        }
    })

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "get_answer":
                question = data.get("question", "")
                await answer_question_stream(question, websocket, user_id)

            elif data.get("type") == "force_question_detection":
                text = data.get("text", "")
                if text:
                    await force_detect_questions(text)

            elif data.get("type") == "clear":
                meeting_state.clear()
                await broadcast({"type": "cleared"})

    except WebSocketDisconnect:
        meeting_state.connected_clients.discard(websocket)
        logger.info(f"Client WebSocket disconnected: user_id={user_id}")

# WebSocket for audio streaming with Deepgram
@app.websocket("/ws/audio")
async def audio_websocket(websocket: WebSocket, token: str = Query(None)):
    await websocket.accept()

    # Get user ID for this connection
    user_id = await get_user_id_from_websocket(websocket, token)
    logger.info(f"Audio WebSocket connected: user_id={user_id}")

    deepgram = Deepgram(os.getenv("DEEPGRAM_API_KEY"))
    dg_connection = None

    # Capture the event loop from the async context
    loop = asyncio.get_running_loop()

    # Track current utterance for interim results
    last_processed = ""

    try:
        # Create a websocket connection to Deepgram
        dg_connection = await deepgram.transcription.live({
            "model": "nova-2",
            "language": "en",
            "encoding": "linear16",
            "sample_rate": 16000,
            "channels": 1,
            "smart_format": True,
            "punctuate": True,
            "diarize": False,
            "interim_results": True,
            "utterance_end_ms": "1000",
            "vad_events": True,
        })

        async def on_message(data):
            nonlocal last_processed

            try:
                # Deepgram SDK v2 returns data in various formats
                # Debug log to understand structure
                # logger.debug(f"Deepgram data type: {type(data)}, data: {data}")

                transcript = ""
                is_final = False

                if isinstance(data, dict):
                    # Check for 'channel' key
                    channel = data.get("channel")
                    if channel:
                        # Channel might be a dict or list
                        if isinstance(channel, dict):
                            alternatives = channel.get("alternatives", [])
                        elif isinstance(channel, list) and len(channel) > 0:
                            alternatives = channel[0].get("alternatives", []) if isinstance(channel[0], dict) else []
                        else:
                            alternatives = []

                        if alternatives and isinstance(alternatives, list) and len(alternatives) > 0:
                            alt = alternatives[0]
                            transcript = alt.get("transcript", "") if isinstance(alt, dict) else getattr(alt, 'transcript', "")

                    is_final = data.get("is_final", False) or data.get("speech_final", False)
                else:
                    # Try to access as object attributes (Deepgram response object)
                    try:
                        if hasattr(data, 'channel'):
                            channel = data.channel
                            if hasattr(channel, 'alternatives') and channel.alternatives:
                                transcript = channel.alternatives[0].transcript
                            elif isinstance(channel, list) and len(channel) > 0:
                                if hasattr(channel[0], 'alternatives') and channel[0].alternatives:
                                    transcript = channel[0].alternatives[0].transcript
                        is_final = getattr(data, 'is_final', False) or getattr(data, 'speech_final', False)
                    except Exception as e:
                        logger.warning(f"Failed to parse Deepgram response: {type(data)}, error: {e}")
                        return

                if not transcript:
                    return

                # No speaker diarization - use single speaker
                speaker = 0

                if is_final:
                    # Final transcript - add to state
                    meeting_state.add_utterance(speaker, transcript)

                    # Check for questions in real-time (keyword-based)
                    question = detect_question(transcript)
                    if question and question != last_processed:
                        last_processed = question
                        q_entry = {
                            "text": question,
                            "speaker": speaker,
                            "timestamp": datetime.now().isoformat(),
                            "source": "keyword"
                        }
                        meeting_state.detected_questions.insert(0, q_entry)
                        meeting_state.detected_questions = meeting_state.detected_questions[:5]

                        # Broadcast question immediately
                        await broadcast({
                            "type": "question_detected",
                            "data": q_entry
                        })

                    # Broadcast transcript update
                    await broadcast({
                        "type": "transcript",
                        "data": {
                            "speaker": speaker,
                            "text": transcript,
                            "is_final": True
                        }
                    })
                else:
                    # Interim result
                    await broadcast({
                        "type": "transcript",
                        "data": {
                            "speaker": speaker,
                            "text": transcript,
                            "is_final": False
                        }
                    })

            except Exception as e:
                logger.error(f"Transcript processing error: {e}")
                import traceback
                traceback.print_exc()

        async def on_error(e, **kwargs):
            logger.error(f"Deepgram error: {e}")

        async def on_close(data=None):
            logger.info("Deepgram connection closed")

        # Register event handlers
        dg_connection.registerHandler(dg_connection.event.TRANSCRIPT_RECEIVED, on_message)
        dg_connection.registerHandler(dg_connection.event.ERROR, on_error)
        dg_connection.registerHandler(dg_connection.event.CLOSE, on_close)

        logger.info("✓ Deepgram connected")

        await websocket.send_json({"type": "ready"})
        logger.info("Sent 'ready' to client, waiting for audio...")

        # Receive and forward audio data to Deepgram
        while True:
            data = await websocket.receive_bytes()
            # SDK v2 send() is not async
            dg_connection.send(data)

    except WebSocketDisconnect:
        logger.info("Audio WebSocket disconnected")
    except Exception as e:
        logger.error(f"Audio WebSocket error: {e}")
    finally:
        if dg_connection:
            # SDK v2 finish() is not async
            dg_connection.finish()

# Document endpoints
@app.get("/api/documents")
async def list_documents(request: Request):
    user_id = await get_user_id_from_request(request)
    store = get_vector_store_for_user(user_id)
    return {"documents": await store.list_documents(user_id), "userId": user_id}

@app.post("/api/documents")
async def upload_document(
    request: Request,
    file: Optional[UploadFile] = File(None),
    content: Optional[str] = Form(None),
    name: Optional[str] = Form(None)
):
    user_id = await get_user_id_from_request(request)
    store = get_vector_store_for_user(user_id)

    if file:
        # Use filename if name not provided
        doc_name = name or file.filename
        file_bytes = await file.read()

        # Extract text from file
        try:
            text_content = extract_text_from_file(file_bytes, doc_name)
        except Exception as e:
            raise HTTPException(400, f"Failed to extract text from file: {str(e)}")

        # Get file type from extension
        file_type = doc_name.lower().split('.')[-1] if '.' in doc_name else 'unknown'

        # Add to Pinecone
        doc = await store.add_document(
            user_id=user_id,
            name=doc_name,
            content=text_content,
            file_name=doc_name,
            file_type=file_type
        )
    elif content:
        if not name:
            raise HTTPException(400, "Name required when uploading text content")
        doc = await store.add_document(
            user_id=user_id,
            name=name,
            content=content,
            file_type='txt'
        )
    else:
        raise HTTPException(400, "No content provided")

    return {
        "id": doc.id,
        "name": doc.name,
        "uploaded_at": doc.uploaded_at.isoformat(),
        "chunks_count": len(doc.chunks),
        "userId": user_id
    }

@app.delete("/api/documents/{doc_id}")
async def delete_document(request: Request, doc_id: str):
    user_id = await get_user_id_from_request(request)
    store = get_vector_store_for_user(user_id)

    if not await store.delete_document(user_id, doc_id):
        raise HTTPException(404, "Document not found")
    return {"success": True}

# State endpoints
@app.get("/api/state")
async def get_state():
    return {
        "transcript": meeting_state.transcript,
        "questions": meeting_state.detected_questions,
        "suggestions": meeting_state.suggestions
    }

@app.post("/api/clear")
async def clear_state():
    meeting_state.clear()
    await broadcast({"type": "cleared"})
    return {"success": True}

# Note: Answers are now streamed via WebSocket only (see /ws/client endpoint)

# Serve frontend
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(
            os.path.join(frontend_dir, "index.html"),
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

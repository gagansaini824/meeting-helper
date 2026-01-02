import os
import json
import asyncio
import logging
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv

from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions
import anthropic

from document_store import document_store
from question_detector import detect_question

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
        logger.info("→ Running Haiku question detection...")
        await detect_questions_with_haiku()

async def global_periodic_suggestions():
    """Generate suggestions every 15 seconds"""
    logger.info("✓ Global periodic suggestions started")
    while True:
        await asyncio.sleep(15)
        await analyze_transcript()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start periodic tasks when app starts
    logger.info("Starting global periodic tasks...")
    question_task = asyncio.create_task(global_periodic_question_detection())
    suggestion_task = asyncio.create_task(global_periodic_suggestions())
    logger.info("✓ Global periodic tasks running")
    yield
    # Cancel tasks on shutdown
    question_task.cancel()
    suggestion_task.cancel()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    """Force question detection on user-selected transcript text"""
    if len(text) < 20:
        return

    try:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        system = """You are a question detector. Analyze the provided text and identify ALL questions being asked.

CRITICAL FORMATTING RULES:
1. ALL questions MUST end with "?" - this is mandatory
2. ALL questions MUST start with a question word: what, how, why, when, where, who, which, can, could, would, should, is, are, do, does, did, has, have, will
3. Extract ALL questions from the text, not just one
4. Combine fragmented questions into complete, grammatically correct questions

Respond with a JSON object containing an array of questions:
{"questions": ["question 1?", "question 2?", "question 3?"]}

If no questions found, return {"questions": []}"""

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            system=system,
            messages=[{"role": "user", "content": f"Selected transcript:\n{text}"}]
        )

        result_text = response.content[0].text.strip()

        # Remove markdown code blocks if present
        if result_text.startswith('```'):
            lines = result_text.split('\n')
            result_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else result_text
            result_text = result_text.replace('```json', '').replace('```', '').strip()

        json_text = result_text.split('\n\n')[0].strip()
        result = json.loads(json_text)
        questions = result.get('questions', [])

        # Add all detected questions
        for question_text in questions:
            if question_text and detect_question(question_text):
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
                    meeting_state.detected_questions = meeting_state.detected_questions[:5]

                    logger.info(f"✓ Manually detected question: {question_text}")
                    # Broadcast immediately
                    await broadcast({
                        "type": "question_detected",
                        "data": q_entry
                    })

    except Exception as e:
        logger.error(f"Force question detection error: {e}")

# Detect questions using Haiku (smarter detection)
async def detect_questions_with_haiku():
    transcript, has_new = meeting_state.get_new_transcript_for_detection(2500)

    # Skip if no new content or too short
    if not has_new or len(transcript) < 50:
        return

    try:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        system = """You are a question detector. Analyze the transcript and identify ALL ACTUAL questions being asked by speakers.

CRITICAL FORMATTING RULES (for regex pattern matching):
1. ALL questions MUST end with "?" - this is mandatory
2. ALL questions MUST start with a question word: what, how, why, when, where, who, which, can, could, would, should, is, are, do, does, did, has, have, will
3. Questions must be at least 10 characters long
4. Combine fragmented questions across multiple utterances into ONE complete, grammatically correct question
5. IMPORTANT: Detect ALL questions in the transcript, including follow-up questions

Valid question formats that will be detected:
✓ "What is the status of the project?"
✓ "How did you implement the RIA and data pipeline?"
✓ "Can you explain how DevOps is implemented in your project?"
✓ "Could you tell me about the architecture?"
✓ "Would you like to discuss this further?"
✓ "Should we proceed with this approach?"

EXAMPLES OF COMBINING FRAGMENTED QUESTIONS:
- Split: "So I would like to know more about how did you implement the" + "RIA and the data pipeline there"
  → Combined: "How did you implement the RIA and the data pipeline?" ✓

- Split: "Can you explain how DevOps" + "is implemented in your project"
  → Combined: "Can you explain how DevOps is implemented in your project?" ✓

INVALID formats to IGNORE (will NOT be detected):
✗ "explain the process" (missing "?" and no question starter)
✗ "Tell me about it" (no "?")
✗ Statements, comments, greetings

Respond with a JSON object containing ALL questions found:
{"questions": ["question 1?", "question 2?", "question 3?"]}

If no questions found, return {"questions": []}"""

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            system=system,
            messages=[{"role": "user", "content": f"Recent transcript:\n{transcript}"}]
        )

        text = response.content[0].text.strip()
        logger.debug(f"Raw Haiku response: {text}")

        # Remove markdown code blocks if present
        if text.startswith('```'):
            # Extract JSON from code block
            lines = text.split('\n')
            text = '\n'.join(lines[1:-1]) if len(lines) > 2 else text
            text = text.replace('```json', '').replace('```', '').strip()

        # Extract only the JSON part (before any explanatory text after double newline)
        # Haiku sometimes adds explanations after the JSON
        json_text = text.split('\n\n')[0].strip()

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
async def analyze_transcript():
    transcript = meeting_state.get_recent_transcript()
    if len(transcript) < 100:
        return

    try:
        # Generate suggestions
        search_results = await document_store.search(transcript, 3)
        doc_context = document_store.get_context(search_results)

        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        doc_section = f"Relevant documents:\n{doc_context}" if doc_context else ""

        system = f"""You are an intelligent meeting assistant. Analyze the conversation transcript and suggest 2-4 highly relevant questions or topics.

{doc_section}

Respond with a JSON array only:
[{{"question": "...", "type": "clarification|context|insight|action"}}]"""

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            system=system,
            messages=[{"role": "user", "content": f"Transcript:\n{transcript}"}]
        )

        text = response.content[0].text
        suggestions = json.loads(text)
        meeting_state.suggestions = suggestions
        await broadcast({"type": "suggestions", "data": suggestions})
    except Exception as e:
        logger.error(f"Analysis error: {e}")

# Answer question with Claude (streaming with proper event handling)
async def answer_question_stream(question: str, websocket: WebSocket):
    full_answer = ""
    try:
        search_results = await document_store.search(question, 5)
        doc_context = document_store.get_context(search_results)

        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        doc_section = f"Available Documents (use ONLY when question is about these specific projects):\n{doc_context}" if doc_context else "No project documents uploaded yet."

        system = f"""You are an interview coach helping someone answer technical interview questions in real-time.

RESPONSE RULES - CRITICAL:
1. **Be BRIEF**: 2-3 sentences MAX for simple questions, 4-5 for complex ones
2. **Start with the answer**: No preamble, no "Great question", just the answer
3. **One simple example**: Include ONE short, concrete example (1-2 lines max)
4. **Interview-ready**: Responses should sound natural when spoken aloud

FORMAT:
- Simple definition/concept → 2-3 sentences + tiny example
- Complex topic → 3-4 bullet points max + example
- Never use headers or lengthy explanations

EXAMPLE RESPONSES:
Q: "What is Docker?"
A: "Docker is a containerization platform that packages applications with their dependencies into lightweight, portable containers. Think of it like a shipping container - everything the app needs travels together. Example: `docker run nginx` spins up a web server in seconds."

Q: "Explain microservices"
A: "Microservices break a large application into small, independent services that communicate via APIs. Each service handles one function and can be deployed separately. Example: An e-commerce app might have separate services for users, orders, and payments."

CONTEXT RULES:
- General knowledge questions → Answer from your knowledge
- Questions about "your project" or specific uploaded docs → Reference the documents below
- Keep examples relevant to the conversation context when possible

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

        # Create stream and iterate chunks
        with client.messages.stream(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1500,
            system=system,
            messages=[{"role": "user", "content": f"Context:\n{transcript}\n\nQuestion: {question}"}]
        ) as stream:
            # Process each text chunk
            for text in stream.text_stream:
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

            logger.info(f"Stream complete. Total chunks: {chunk_count}")
            # Get final message
            final_message = stream.get_final_message()

        # Send final message with complete answer
        await websocket.send_json({
            "type": "answer_chunk",
            "data": {
                "question": question,
                "chunk": "",
                "is_final": True,
                "full_answer": full_answer,
                "stop_reason": final_message.stop_reason
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
async def client_websocket(websocket: WebSocket):
    await websocket.accept()
    meeting_state.connected_clients.add(websocket)
    
    # Send current state
    await websocket.send_json({
        "type": "init",
        "data": {
            "transcript": meeting_state.transcript,
            "questions": meeting_state.detected_questions,
            "suggestions": meeting_state.suggestions
        }
    })
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "get_answer":
                question = data.get("question", "")
                await answer_question_stream(question, websocket)

            elif data.get("type") == "force_question_detection":
                text = data.get("text", "")
                if text:
                    await force_detect_questions(text)

            elif data.get("type") == "clear":
                meeting_state.clear()
                await broadcast({"type": "cleared"})
    
    except WebSocketDisconnect:
        meeting_state.connected_clients.discard(websocket)

# WebSocket for audio streaming with Deepgram
@app.websocket("/ws/audio")
async def audio_websocket(websocket: WebSocket):
    await websocket.accept()

    deepgram = DeepgramClient(api_key=os.getenv("DEEPGRAM_API_KEY"))
    dg_connection = None

    # Capture the event loop from the async context
    loop = asyncio.get_running_loop()

    try:
        # Configure Deepgram live transcription
        options = LiveOptions(
            model="nova-2",
            language="en",
            encoding="linear16",
            sample_rate=16000,
            channels=1,
            smart_format=True,
            punctuate=True,
            diarize=False,  # Disable speaker diarization
            interim_results=True,
            utterance_end_ms=1000,
            vad_events=True,
        )

        dg_connection = deepgram.listen.websocket.v("1")

        # Track current utterance for interim results
        last_processed = ""

        def on_message(self, result, **kwargs):
            nonlocal last_processed

            try:
                transcript = result.channel.alternatives[0].transcript
                if not transcript:
                    return

                is_final = result.is_final

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
                        loop.create_task(broadcast({
                            "type": "question_detected",
                            "data": q_entry
                        }))

                    # Broadcast transcript update
                    loop.create_task(broadcast({
                        "type": "transcript",
                        "data": {
                            "speaker": speaker,
                            "text": transcript,
                            "is_final": True
                        }
                    }))
                else:
                    # Interim result
                    loop.create_task(broadcast({
                        "type": "transcript",
                        "data": {
                            "speaker": speaker,
                            "text": transcript,
                            "is_final": False
                        }
                    }))

            except Exception as e:
                logger.error(f"Transcript processing error: {e}")
                import traceback
                traceback.print_exc()

        def on_error(self, error, **kwargs):
            logger.error(f"Deepgram error: {error}")

        logger.info("Setting up Deepgram event handlers...")
        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        dg_connection.on(LiveTranscriptionEvents.Error, on_error)

        logger.info("Starting Deepgram connection...")
        if not dg_connection.start(options):
            raise Exception("Failed to start Deepgram connection")

        logger.info("✓ Deepgram connected")

        await websocket.send_json({"type": "ready"})
        logger.info("Sent 'ready' to client, waiting for audio...")
        
        # Receive and forward audio data to Deepgram
        while True:
            data = await websocket.receive_bytes()
            dg_connection.send(data)
    
    except WebSocketDisconnect:
        logger.info("Audio WebSocket disconnected")
    except Exception as e:
        logger.error(f"Audio WebSocket error: {e}")
    finally:
        if dg_connection:
            dg_connection.finish()

# Document endpoints
@app.get("/api/documents")
async def list_documents():
    return {"documents": document_store.list_documents()}

@app.post("/api/documents")
async def upload_document(
    file: Optional[UploadFile] = File(None),
    content: Optional[str] = Form(None),
    name: Optional[str] = Form(None)
):
    if file:
        # Use filename if name not provided
        doc_name = name or file.filename
        file_bytes = await file.read()

        # Pass file bytes to extract text based on file type
        doc = await document_store.add_document(doc_name, "", file_bytes=file_bytes)
    elif content:
        if not name:
            raise HTTPException(400, "Name required when uploading text content")
        doc = await document_store.add_document(name, content)
    else:
        raise HTTPException(400, "No content provided")

    return {
        "id": doc.id,
        "name": doc.name,
        "uploaded_at": doc.uploaded_at.isoformat(),
        "chunks_count": len(doc.chunks)
    }

@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    if not document_store.remove_document(doc_id):
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
        return FileResponse(os.path.join(frontend_dir, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

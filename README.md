# Meeting Assistant

Real-time AI meeting companion with speaker diarization.

## Features

- **Real-time transcription** with Deepgram Nova-2
- **Speaker diarization** - identifies who is speaking (Speaker 0, 1, 2...)
- **Instant question detection** - questions highlighted immediately
- **AI-powered answers** - Claude with web search + your documents
- **Periodic suggestions** - context-aware topic suggestions every 15s
- **Document context** - upload docs for enhanced answers

## Setup

1. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

2. Create `.env` file:
```bash
cp .env.example .env
# Edit .env with your API keys:
# - DEEPGRAM_API_KEY (get from console.deepgram.com)
# - ANTHROPIC_API_KEY (get from console.anthropic.com)
# - OPENAI_API_KEY (for document embeddings)
```

3. Run the server:
```bash
python main.py
```

4. Open http://localhost:8000

## Usage

1. Click "Start Listening"
2. Allow microphone access
3. Keep your meeting audio playing through speakers
4. Watch for detected questions and click for answers
5. Upload documents for context-aware responses

## Architecture

```
Browser Microphone
      │
      ▼ (WebSocket: audio bytes)
FastAPI Backend
      │
      ├─► Deepgram (real-time transcription + diarization)
      │       │
      │       ▼ (speaker-labeled transcript)
      │   Question Detector (instant)
      │       │
      │       ▼
      │   Broadcast to clients
      │
      ├─► Every 15s: Claude analyzes transcript → suggestions
      │
      └─► On click: Claude + web search → answer
```

## API Keys Required

| Service | Purpose | Get Key |
|---------|---------|---------|
| Deepgram | Real-time transcription | console.deepgram.com |
| Anthropic | Analysis & answers | console.anthropic.com |
| OpenAI | Document embeddings | platform.openai.com |

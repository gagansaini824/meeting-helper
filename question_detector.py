import re

QUESTION_PATTERNS = [
    re.compile(r'\b(what|how|why|when|where|who|which)\s+.{5,}\?$', re.IGNORECASE),
    re.compile(r'\b(can|could|would|should|is|are|do|does|did|has|have|will)\s+.{5,}\?$', re.IGNORECASE),
    re.compile(r'\b(explain|clarify|elaborate|tell me about|describe)\s+.{3,}', re.IGNORECASE),
    re.compile(r'\b(does anyone know|any idea|any thoughts on|what do you think)\b.{3,}', re.IGNORECASE),
    re.compile(r"\b(what's the|how do we|can we|should we|what about)\s+.{3,}", re.IGNORECASE),
    re.compile(r'\b(is there|are there|have you|do you know)\s+.{3,}\??$', re.IGNORECASE),
]

QUESTION_STARTERS = [
    'what', 'how', 'why', 'when', 'where', 'who', 'which',
    'can', 'could', 'would', 'should', 'is', 'are', 'do', 'does', 'did',
    'has', 'have', 'will', 'shall'
]

def detect_question(text: str) -> str | None:
    if not text or len(text) < 10:
        return None
    
    sentences = re.split(r'[.!]\s+', text)
    last_sentence = sentences[-1].strip() if sentences else ""
    
    if not last_sentence or len(last_sentence) < 10:
        return None
    
    # Direct question mark
    if last_sentence.endswith('?'):
        return last_sentence
    
    # Pattern matching
    for pattern in QUESTION_PATTERNS:
        if pattern.search(last_sentence):
            return last_sentence
    
    # Starts with question word
    first_word = last_sentence.split()[0].lower() if last_sentence.split() else ""
    if first_word in QUESTION_STARTERS and len(last_sentence) > 15:
        return last_sentence
    
    return None

def extract_questions(transcript: str, limit: int = 5) -> list[str]:
    sentences = re.split(r'[.!?]\s+', transcript)
    questions = []
    
    for i in range(len(sentences) - 1, -1, -1):
        if len(questions) >= limit:
            break
        sentence = sentences[i].strip()
        if sentence and detect_question(sentence + '?'):
            q = sentence if sentence.endswith('?') else sentence + '?'
            questions.append(q)
    
    return questions

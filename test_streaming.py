import os
import asyncio
from dotenv import load_dotenv
import anthropic

load_dotenv()

async def test_streaming():
    print("Testing Claude streaming...")
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    print("\nStarting stream...")
    chunk_count = 0
    full_text = ""

    with client.messages.stream(
        model="claude-sonnet-4-5-20250929",
        max_tokens=500,
        messages=[{"role": "user", "content": "Explain what recall is in machine learning in 2 sentences."}]
    ) as stream:
        for text in stream.text_stream:
            chunk_count += 1
            full_text += text
            print(f"Chunk {chunk_count}: '{text}' ({len(text)} chars)")

    print(f"\n✓ Stream complete!")
    print(f"Total chunks: {chunk_count}")
    print(f"Full text: {full_text}")

if __name__ == "__main__":
    asyncio.run(test_streaming())

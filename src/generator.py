import os
import logging
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Check your .env file.")

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.5-flash")


def build_prompt(query: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join([
        f"[Source: {c['source']}]\n{c['text']}"
        for c in chunks
    ])

    return f"""You are a helpful study assistant. Answer the question using ONLY the context provided.
If the answer is not in the context, say "I couldn't find that in your uploaded documents."
Always mention which document your answer came from.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""


def generate_answer(query: str, chunks: list[dict]):
    prompt = build_prompt(query, chunks)

    try:
        response = model.generate_content(prompt, stream=True)

        for chunk in response:
            if chunk.text:
                yield chunk.text

    except Exception:
        logger.exception("Generation error")
        raise ValueError("Unable to generate an answer at this time.")
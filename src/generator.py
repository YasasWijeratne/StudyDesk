import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

def build_prompt(query: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join([
        f"[Source: {c['source']}]\n{c['text']}"
        for c in chunks
    ])

    prompt = f"""You are a helpful study assistant. A student has uploaded their documents and is asking questions about them.
Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I couldn't find that in your uploaded documents."
Always mention which document your answer came from.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""

    return prompt


def generate_answer(query: str, chunks: list[dict]):
    prompt = build_prompt(query, chunks)
    response = model.generate_content(prompt, stream=True)

    for chunk in response:
        if chunk.text:
            yield chunk.text
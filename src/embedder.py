import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

def embed_chunks(chunks: list[str]) -> list[list[float]]:
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=chunks,
        task_type="retrieval_document"
    )
    return result["embedding"]


def embed_query(query: str) -> list[float]:
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=query,
        task_type="retrieval_query"
    )
    return result["embedding"]
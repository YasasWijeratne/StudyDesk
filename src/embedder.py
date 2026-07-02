import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found.")

genai.configure(api_key=api_key)


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    try:
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=chunks,
            task_type="retrieval_document"
        )

        embeddings = result["embedding"]

        if len(embeddings) != len(chunks):
            raise ValueError("Embedding mismatch.")

        return embeddings

    except Exception:
        raise ValueError("Failed to generate embeddings.")


def embed_query(query: str) -> list[float]:
    try:
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=query,
            task_type="retrieval_query"
        )
        return result["embedding"]

    except Exception:
        raise ValueError("Failed to generate query embedding.")
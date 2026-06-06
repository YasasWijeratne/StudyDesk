import chromadb
from chromadb.config import Settings

def get_client():
    return chromadb.EphemeralClient()

def get_or_create_collection(client, session_id: str):
    return client.get_or_create_collection(
        name=f"session_{session_id}",
        metadata={"hnsw:space": "cosine"}
    )

def store_chunks(collection, chunks: list[str], embeddings: list[list[float]], file_name: str):
    collection.add(
        ids=[f"{file_name}_chunk_{i}" for i in range(len(chunks))],
        embeddings=embeddings,
        documents=chunks,
        metadatas=[{"source": file_name, "chunk_index": i} for i in range(len(chunks))]
    )

def delete_collection(client, session_id: str):
    try:
        client.delete_collection(f"session_{session_id}")
    except:
        pass
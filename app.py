import streamlit as st
import uuid
from src.parser import parse_file
from src.chunker import chunk_text
from src.embedder import embed_chunks
from src.vectorstore import get_client, get_or_create_collection, store_chunks, delete_collection
from src.retriever import retrieve_chunks
from src.generator import generate_answer

# Page config 
st.set_page_config(
    page_title="StudyDesk",
    page_icon="📚",
    layout="centered"
)

# Session state init 
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = get_client()

if "collection" not in st.session_state:
    st.session_state.collection = get_or_create_collection(
        st.session_state.chroma_client,
        st.session_state.session_id
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# UI
st.title("📚 StudyDesk")
st.caption("Upload your documents and ask anything about them.")

# File upload
uploaded_file = st.file_uploader(
    "Upload a PDF or TXT file",
    type=["pdf", "txt"],
    accept_multiple_files=False
)

if uploaded_file is not None:
    if uploaded_file.name not in st.session_state.uploaded_files:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            file_bytes = uploaded_file.read()

            # Parse
            success, result = parse_file(uploaded_file.name, file_bytes)

            if not success:
                st.error(result)
            else:
                # Chunk
                chunks = chunk_text(result)

                # Embed
                embeddings = embed_chunks(chunks)

                # Store
                store_chunks(
                    st.session_state.collection,
                    chunks,
                    embeddings,
                    uploaded_file.name
                )

                st.session_state.uploaded_files.append(uploaded_file.name)
                st.success(f"✅ {uploaded_file.name} indexed — {len(chunks)} chunks")

# Show uploaded files
if st.session_state.uploaded_files:
    st.info("📄 Uploaded: " + ", ".join(st.session_state.uploaded_files))

st.divider()

# Chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input 
query = st.chat_input("Ask something about your documents...")

if query:
    if not st.session_state.uploaded_files:
        st.warning("Please upload a document first.")
    else:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Retrieve and generate
        with st.chat_message("assistant"):
            chunks = retrieve_chunks(st.session_state.collection, query)

            if not chunks:
                response = "I couldn't find anything relevant in your documents."
                st.markdown(response)
            else:
                response = st.write_stream(generate_answer(query, chunks))

        st.session_state.messages.append({"role": "assistant", "content": response})
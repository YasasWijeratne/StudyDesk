import streamlit as st
import uuid
from src.parser import parse_file, get_safe_filename
from src.chunker import chunk_text
from src.embedder import embed_chunks
from src.vectorstore import get_client, get_or_create_collection, store_chunks, delete_collection
from src.retriever import retrieve_chunks
from src.generator import generate_answer


def load_css(path: str):
    with open(path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Page config 
st.set_page_config(
    page_title="StudyDesk",
    page_icon="📚",
    layout="centered"
)

load_css("assets/style.css")

# Session state 
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

# Header 
st.markdown("""
    <div class="app-header">
        <div class="app-title">📚 StudyDesk</div>
        <div class="app-subtitle">Upload your documents and ask anything about them</div>
    </div>
""", unsafe_allow_html=True)

# File upload 
st.markdown('<div class="upload-label">Documents</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a PDF — max 10MB",
    type=["pdf"],
    accept_multiple_files=False,
    label_visibility="collapsed"
)

if uploaded_file is not None:
    safe_name = get_safe_filename(uploaded_file.name)

    if safe_name not in st.session_state.uploaded_files:
        file_bytes = uploaded_file.read()

        with st.spinner(f"Processing {safe_name}..."):
            success, result = parse_file(safe_name, file_bytes)

            if not success:
                st.error(result)
            else:
                chunks = chunk_text(result)
                embeddings = embed_chunks(chunks)
                store_chunks(
                    st.session_state.collection,
                    chunks,
                    embeddings,
                    safe_name
                )
                st.session_state.uploaded_files.append(safe_name)
                st.success(f"Indexed {len(chunks)} chunks from {safe_name}")

# File pills
if st.session_state.uploaded_files:
    pills_html = "".join([
        f'<span class="file-pill">📄 {f}</span>'
        for f in st.session_state.uploaded_files
    ])
    st.markdown(
        f'<div style="margin: 0.5rem 0 1.5rem 0">{pills_html}</div>',
        unsafe_allow_html=True
    )

st.divider()

# Empty states 
if not st.session_state.messages and not st.session_state.uploaded_files:
    st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">📖</div>
            <div class="empty-state-text">Upload a document above to get started</div>
        </div>
    """, unsafe_allow_html=True)
elif not st.session_state.messages and st.session_state.uploaded_files:
    st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">💬</div>
            <div class="empty-state-text">Document ready — ask your first question below</div>
        </div>
    """, unsafe_allow_html=True)

# Chat history 
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input 
query = st.chat_input("Ask something about your documents...")

if query:
    if not st.session_state.uploaded_files:
        st.warning("Upload a document first before asking questions.")
    else:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            try:
                chunks = retrieve_chunks(st.session_state.collection, query)

                if not chunks:
                    response = "I couldn't find anything relevant in your documents."
                    st.markdown(response)
                else:
                    response = st.write_stream(generate_answer(query, chunks))

            except ValueError as error:
                response = "Something went wrong while generating the answer."
                st.error(response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })
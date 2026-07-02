import uuid
import streamlit as st

from src.parser import parse_file, get_safe_filename
from src.chunker import chunk_text
from src.embedder import embed_chunks
from src.vectorstore import (
    get_client,
    get_or_create_collection,
    store_chunks,
)
from src.retriever import retrieve_chunks
from src.generator import generate_answer


st.set_page_config(
    page_title="StudyDesk",
    page_icon="📚",
    layout="wide"
)


def load_css(path: str):
    with open(path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def safe_error(message: str = "Something went wrong. Please try again."):
    st.error(message)


load_css("assets/style.css")


# 
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


with st.sidebar:
    st.markdown("## StudyDesk")

    if st.button("➕ New Chat"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.collection = get_or_create_collection(
            st.session_state.chroma_client,
            st.session_state.session_id
        )
        st.rerun()

    st.divider()

    st.markdown("### Documents")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        safe_name = get_safe_filename(uploaded_file.name)

        if safe_name not in st.session_state.uploaded_files:
            file_bytes = uploaded_file.read()

            with st.spinner("Processing document..."):
                success, result = parse_file(safe_name, file_bytes)

                if not success:
                    safe_error(result)
                else:
                    try:
                        chunks = chunk_text(result)
                        embeddings = embed_chunks(chunks)

                        store_chunks(
                            st.session_state.collection,
                            chunks,
                            embeddings,
                            safe_name
                        )

                        st.session_state.uploaded_files.append(safe_name)

                        st.success("File indexed successfully")

                    except ValueError:
                        safe_error("Failed to process document.")
                    except Exception:
                        safe_error("Unexpected error during indexing.")

    st.divider()

    if st.session_state.uploaded_files:
        st.markdown("### Uploaded Files")
        for f in st.session_state.uploaded_files:
            st.markdown(f" {f}")


st.markdown(
    """
    <div class="chat-header">
        <h2>Chat</h2>
    </div>
    """,
    unsafe_allow_html=True
)


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


query = st.chat_input("Ask something about your documents...")


if query:
    if not st.session_state.uploaded_files:
        st.warning("Please upload a document first.")
    else:
        st.session_state.messages.append(
            {"role": "user", "content": query}
        )

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            try:
                chunks = retrieve_chunks(
                    st.session_state.collection,
                    query
                )

                if not chunks:
                    response = "I couldn't find relevant information in your documents."
                    st.markdown(response)

                else:
                    response = st.write_stream(
                        generate_answer(query, chunks)
                    )

            except ValueError:
                response = "Something went wrong while processing your request."
                safe_error(response)

            except Exception:
                response = "Unexpected error occurred. Please try again."
                safe_error(response)

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )
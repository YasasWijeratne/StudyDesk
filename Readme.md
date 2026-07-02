# StudyDesk

A RAG app where you can upload your documents and ask questions about them. Built from scratch without LangChain or any other abstraction framework.

---

## What it does

You upload a PDF or TXT file and chat with it. The app breaks the document into chunks, embeds them, and stores them in a vector database. When you ask something, it finds the most relevant chunks and sends them to the LLM to generate an answer — and tells you which document the answer came from.

---

## Stack

| Component | Tool |
|---|---|
| LLM | Gemini 2.5 Flash |
| Embeddings | Gemini Embedding 001 |
| Vector Database | ChromaDB (in-memory) |
| PDF Parsing | PyMuPDF |
| UI | Streamlit |

---

## How it works

When a file is uploaded:
```
Upload → Parse → Chunk → Embed → Store in ChromaDB
```

When a question is asked:
```
Question → Embed → Find top 4 similar chunks → Build prompt → Stream answer
```

---

## Running it locally

```bash
git clone https://github.com/YasasWijeratne/StudyDesk.git
cd StudyDesk

python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
streamlit run app.py
```

---

## Project structure

```
StudyDesk/
├── src/
│   ├── parser.py        
│   ├── chunker.py       
│   ├── embedder.py      
│   ├── vectorstore.py   
│   ├── retriever.py     
│   └── generator.py     
├── assets/
│   └── style.css        
├── app.py               
└── requirements.txt
```

---

## Notes

- Max file size is 10MB
- Supported formats: PDF
- Documents are stored in memory only — nothing persists after the session ends

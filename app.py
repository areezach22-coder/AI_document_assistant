import os
import re
import hashlib
import tempfile
from pathlib import Path

import faiss
import gdown
import numpy as np
import streamlit as st
from docx import Document
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄",
    layout="wide",
)

st.title("📄 AI Document Assistant")
st.write(
    "Upload documents or load public Google Drive files, then ask questions "
    "using hybrid semantic + keyword search."
)

SUPPORTED_TYPES = ["pdf", "docx", "txt", "md"]


# -----------------------------
# Session state
# -----------------------------
if "documents" not in st.session_state:
    st.session_state.documents = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "processed_ids" not in st.session_state:
    st.session_state.processed_ids = set()


# -----------------------------
# Cached embedding model
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# Document extraction
# -----------------------------
def extract_pdf(file_path):
    """Extract PDF text while preserving page numbers."""
    reader = PdfReader(file_path)
    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            documents.append(
                {
                    "filename": Path(file_path).name,
                    "page": page_number,
                    "text": text.strip(),
                }
            )

    return documents


def extract_docx(file_path):
    """Extract DOCX paragraphs. DOCX does not reliably expose page numbers."""
    document = Document(file_path)
    text = "\n".join(
        paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()
    )

    if not text.strip():
        return []

    return [
        {
            "filename": Path(file_path).name,
            "page": None,
            "text": text.strip(),
        }
    ]


def extract_txt(file_path):
    """Extract plain TXT text."""
    text = Path(file_path).read_text(encoding="utf-8", errors="ignore")

    if not text.strip():
        return []

    return [
        {
            "filename": Path(file_path).name,
            "page": None,
            "text": text.strip(),
        }
    ]


def extract_md(file_path):
    """Extract Markdown text as plain text."""
    text = Path(file_path).read_text(encoding="utf-8", errors="ignore")

    if not text.strip():
        return []

    return [
        {
            "filename": Path(file_path).name,
            "page": None,
            "text": text.strip(),
        }
    ]


def extract_document(file_path):
    """Choose the correct extractor from the file extension."""
    extension = Path(file_path).suffix.lower()

    if extension == ".pdf":
        return extract_pdf(file_path)
    if extension == ".docx":
        return extract_docx(file_path)
    if extension == ".txt":
        return extract_txt(file_path)
    if extension == ".md":
        return extract_md(file_path)

    return []


# -----------------------------
# Text chunking
# -----------------------------
def chunk_text(text, chunk_size=700, overlap=120):
    """Create overlapping word-based chunks."""
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))

        if end == len(words):
            break

        start = end - overlap

    return chunks


def create_chunks(extracted_documents):
    """Create chunks while preserving filename and page metadata."""
    all_chunks = []

    for document in extracted_documents:
        pieces = chunk_text(document["text"])

        for piece in pieces:
            all_chunks.append(
                {
                    "filename": document["filename"],
                    "page": document["page"],
                    "text": piece,
                }
            )

    return all_chunks


# -----------------------------
# Embeddings + FAISS
# -----------------------------
def build_vector_store(chunks):
    """Create embeddings once and store them in a FAISS index."""
    if not chunks:
        return None, None

    model = load_embedding_model()
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return embeddings, index


# -----------------------------
# Keyword search
# -----------------------------
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "what", "which",
    "who", "when", "where", "why", "how", "of", "to", "in", "on",
    "for", "and", "or", "with", "from", "this", "that", "it", "about",
    "can", "could", "would", "should", "do", "does", "did", "tell",
    "me", "please", "explain",
}


def important_words(text):
    words = re.findall(r"\b[a-zA-Z0-9][a-zA-Z0-9_-]*\b", text.lower())
    return [word for word in words if word not in STOP_WORDS and len(word) > 2]


def keyword_scores(question, chunks):
    """Score chunks by the fraction of important query words they contain."""
    query_words = set(important_words(question))
    scores = []

    for chunk in chunks:
        chunk_words = set(important_words(chunk["text"]))
        if not query_words:
            scores.append(0.0)
        else:
            scores.append(len(query_words & chunk_words) / len(query_words))

    return np.array(scores, dtype="float32")


# -----------------------------
# Hybrid search
# -----------------------------
def hybrid_search(question, top_k=5):
    """Combine FAISS semantic similarity and keyword matching."""
    if not st.session_state.chunks or st.session_state.faiss_index is None:
        return []

    model = load_embedding_model()

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True,
    ).astype("float32")

    semantic_scores, semantic_ids = st.session_state.faiss_index.search(
        question_embedding, len(st.session_state.chunks)
    )

    semantic_scores = semantic_scores[0]
    semantic_ids = semantic_ids[0]

    # Normalize cosine/IP scores into approximately 0-1.
    semantic_scores = np.clip((semantic_scores + 1.0) / 2.0, 0.0, 1.0)

    keyword = keyword_scores(question, st.session_state.chunks)

    combined = []

    for rank, chunk_id in enumerate(semantic_ids):
        if chunk_id < 0:
            continue

        score = (0.70 * float(semantic_scores[rank])) + (
            0.30 * float(keyword[chunk_id])
        )

        combined.append(
            {
                "score": score,
                "semantic_score": float(semantic_scores[rank]),
                "keyword_score": float(keyword[chunk_id]),
                "chunk": st.session_state.chunks[chunk_id],
            }
        )

    combined.sort(key=lambda item: item["score"], reverse=True)

    return combined[:top_k]


# -----------------------------
# Groq
# -----------------------------
def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

    if not api_key:
        return None

    return Groq(api_key=api_key)


def answer_question(question, retrieved_chunks):
    """Ask Groq to answer only from retrieved context."""
    client = get_groq_client()

    if client is None:
        return (
            "GROQ_API_KEY is not configured. Add it to Streamlit secrets "
            "as GROQ_API_KEY."
        )

    context_parts = []

    for item in retrieved_chunks:
        chunk = item["chunk"]
        page_text = f", page {chunk['page']}" if chunk["page"] else ""
        context_parts.append(
            f"Source: {chunk['filename']}{page_text}\n"
            f"Text: {chunk['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    system_prompt = """You are an AI document assistant.

Answer the user's question ONLY using the provided document context.
Do not use outside knowledge.
If the answer is not present in the context, clearly say:
"The information is not available in the provided documents."

Keep the answer clear and concise.
"""

    user_prompt = f"""Document context:

{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    return response.choices[0].message.content


# -----------------------------
# Google Drive
# -----------------------------
# -----------------------------
# Google Drive
# -----------------------------
def download_drive_source(url):
    """Download a single public Google Drive file."""

    temp_dir = Path(
        tempfile.mkdtemp(prefix="document_assistant_drive_")
    )

    # Extract Google Drive file ID
    match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)

    if not match:
        raise ValueError("Invalid Google Drive file link.")

    file_id = match.group(1)

    # Download the file
    output_file = temp_dir / "drive_file"

    downloaded = gdown.download(
        id=file_id,
        output=str(output_file),
        quiet=False,
    )

    if not downloaded or not Path(downloaded).is_file():
        return []

    downloaded_path = Path(downloaded)

    # Check file type
    file_signature = downloaded_path.read_bytes()[:10]

    # PDF
    if file_signature.startswith(b"%PDF"):
        final_path = downloaded_path.with_suffix(".pdf")

    # DOCX
    elif file_signature.startswith(b"PK"):
        final_path = downloaded_path.with_suffix(".docx")

    # TXT / MD
    else:
        final_path = downloaded_path.with_suffix(".txt")

    downloaded_path.rename(final_path)

    return [final_path]
    # -----------------------------
    # Google Drive folder
    # -----------------------------
    if "/folders/" in url:
        downloaded_files = gdown.download_folder(
            url,
            output=str(temp_dir),
            quiet=False,
            remaining_ok=True,
        )

        if not downloaded_files:
            return []

        return [
            Path(file_path)
            for file_path in downloaded_files
            if Path(file_path).is_file()
        ]

    # -----------------------------
    # Google Drive single file
    # -----------------------------
    output_file = temp_dir / "drive_file"

    downloaded = gdown.download(
        url=url,
        output=str(output_file),
        quiet=False,
    )

    if downloaded and Path(downloaded).is_file():
        return [Path(downloaded)]

    return []

def load_files_into_app(file_paths):
    """Extract, chunk and embed new files only."""
    new_documents = []
    new_chunks = []

    for file_path in file_paths:
        file_path = Path(file_path)

        if file_path.suffix.lower().lstrip(".") not in SUPPORTED_TYPES:
            continue

        file_bytes = file_path.read_bytes()
        file_id = hashlib.sha256(file_bytes).hexdigest()

        if file_id in st.session_state.processed_ids:
            continue

        extracted = extract_document(file_path)

        if extracted:
            new_documents.extend(extracted)
            chunks = create_chunks(extracted)
            new_chunks.extend(chunks)
            st.session_state.processed_ids.add(file_id)

    if not new_chunks:
        return 0, 0

    # Rebuild the complete vector store only when new documents arrive.
    st.session_state.documents.extend(new_documents)
    st.session_state.chunks.extend(new_chunks)

    embeddings, index = build_vector_store(st.session_state.chunks)
    st.session_state.embeddings = embeddings
    st.session_state.faiss_index = index

    return len(new_documents), len(new_chunks)


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Document Sources")

    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT or MD files",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
    )

    if st.button("Process uploaded files", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one document.")
        else:
            temp_files = []

            for uploaded_file in uploaded_files:
                suffix = Path(uploaded_file.name).suffix
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix,
                )
                temp_file.write(uploaded_file.getvalue())
                temp_file.close()
                temp_files.append(Path(temp_file.name))

                # Keep the original filename by renaming the temporary file.
                original_name_file = Path(temp_file.name).with_name(
                    uploaded_file.name
                )
                Path(temp_file.name).rename(original_name_file)
                temp_files[-1] = original_name_file

            doc_count, chunk_count = load_files_into_app(temp_files)

            if doc_count:
                st.success(
                    f"Processed {doc_count} document section(s) and "
                    f"created {chunk_count} new chunks."
                )
            else:
                st.info("No new supported documents were added.")

    st.divider()

    st.subheader("Google Drive")
    drive_url = st.text_input(
        "Public Drive file or folder link",
        placeholder="Paste Google Drive link here",
    )

    if st.button("Load from Google Drive", use_container_width=True):
        if not drive_url.strip():
            st.warning("Please paste a Google Drive link.")
        else:
            with st.spinner("Downloading Drive files..."):
                try:
                    drive_files = download_drive_source(drive_url.strip())

                    if not drive_files:
                        st.error(
                            "No files were loaded. Make sure the Drive file/folder "
                            "is publicly accessible and contains PDF, DOCX, TXT or MD files."
                        )
                    else:
                        doc_count, chunk_count = load_files_into_app(drive_files)
                        st.success(
                            f"Loaded {doc_count} document section(s) and "
                            f"created {chunk_count} new chunks."
                        )
                except Exception as exc:
                    st.error(f"Google Drive loading failed: {exc}")

    st.divider()

    st.subheader("Current index")
    st.write(f"Document sections: {len(st.session_state.documents)}")
    st.write(f"Chunks: {len(st.session_state.chunks)}")

    if st.button("Clear all documents", use_container_width=True):
        st.session_state.documents = []
        st.session_state.chunks = []
        st.session_state.embeddings = None
        st.session_state.faiss_index = None
        st.session_state.processed_ids = set()
        st.rerun()


# -----------------------------
# Main document information
# -----------------------------
st.subheader("Extracted Document Information")

if st.session_state.documents:
    for document in st.session_state.documents:
        page = document["page"] if document["page"] else "N/A"
        preview = document["text"][:300].replace("\n", " ")

        st.markdown(
            f"**Filename:** {document['filename']}  \n"
            f"**Page:** {page}  \n"
            f"**Text preview:** {preview}..."
        )
        st.divider()
else:
    st.info("Upload a document or load a public Google Drive file/folder to begin.")


# -----------------------------
# Question answering
# -----------------------------
st.subheader("Ask a Question")

question = st.text_input(
    "Enter your question",
    placeholder="What is this document about?",
)

top_k = st.slider(
    "Number of sources to retrieve",
    min_value=1,
    max_value=8,
    value=5,
)

if st.button("Ask", type="primary"):
    if not st.session_state.chunks:
        st.warning("Please add documents first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching documents and generating answer..."):
            results = hybrid_search(question, top_k=top_k)

            if not results:
                st.warning("No relevant document chunks were found.")
            else:
                answer = answer_question(question, results)

                st.markdown("### Answer")
                st.write(answer)

                st.markdown("### Retrieved Sources")

                for number, result in enumerate(results, start=1):
                    chunk = result["chunk"]
                    page = chunk["page"] if chunk["page"] else "N/A"

                    with st.expander(
                        f"{number}. {chunk['filename']} | Page: {page} | "
                        f"Hybrid score: {result['score']:.3f}"
                    ):
                        st.write(
                            f"**Filename:** {chunk['filename']}  \n"
                            f"**Page:** {page}  \n"
                            f"**Semantic score:** {result['semantic_score']:.3f}  \n"
                            f"**Keyword score:** {result['keyword_score']:.3f}"
                        )
                        st.write(chunk["text"])

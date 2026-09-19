# AI Document Assistant

A simple Streamlit RAG-style document assistant.

It supports:

- PDF
- DOCX
- TXT
- Markdown (MD)
- Public Google Drive file/folder links
- Text extraction
- Overlapping text chunking
- Sentence Transformers embeddings
- FAISS semantic search
- Keyword search
- Hybrid search
- Groq question answering
- Retrieved source display
- Streamlit session-state caching so embeddings are not recreated for every question

## Project files

```text
document_assistant/
│
├── app.py
├── requirements.txt
└── README.md
```

## 1. Install requirements

Create a virtual environment if you want, then run:

```bash
pip install -r requirements.txt
```

## 2. Add the Groq API key

Do NOT put the API key directly in `app.py`.

For Streamlit Community Cloud:

1. Open your deployed app.
2. Go to **Settings > Secrets**.
3. Add:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

For local Streamlit, create:

```text
.streamlit/secrets.toml
```

and add:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

## 3. Run the app

```bash
streamlit run app.py
```

## How the app works

### Step 1: Document extraction

Each file type has its own function:

- `extract_pdf()`
- `extract_docx()`
- `extract_txt()`
- `extract_md()`

PDF extraction keeps the page number.

DOCX, TXT and MD normally do not have reliable page information, so their page value is stored as `None`.

### Step 2: Chunking

Long extracted text is divided into smaller word-based chunks.

The default settings are:

- Chunk size: 700 words
- Overlap: 120 words

The overlap helps preserve context between neighboring chunks.

### Step 3: Embeddings

The app uses:

```text
all-MiniLM-L6-v2
```

from Sentence Transformers.

Every chunk receives a vector embedding.

The embeddings are stored in Streamlit session state so they can be reused for later questions during the session.

### Step 4: FAISS

FAISS stores the chunk embeddings and performs semantic similarity search.

The question is embedded with the same Sentence Transformer model.

FAISS then finds chunks that are semantically similar to the question.

### Step 5: Keyword search

The app also checks important words from the question against each chunk.

This helps when an exact technical term appears in a document but semantic search does not rank it highly enough.

### Step 6: Hybrid search

The final score is:

```text
Hybrid Score =
0.70 × Semantic Score +
0.30 × Keyword Score
```

The highest-scoring chunks are passed to the LLM.

### Step 7: Groq

Groq receives:

1. The user's question
2. The retrieved document chunks

The system prompt instructs the model to answer only from the supplied context.

If the answer is not in the retrieved context, it should say:

```text
The information is not available in the provided documents.
```

### Step 8: Sources

After every answer, the app shows:

- Filename
- Page number when available
- Semantic score
- Keyword score
- Hybrid score
- Retrieved text chunk

## Google Drive

The Drive loader uses `gdown`.

The Drive file or folder must be publicly accessible.

Supported file types are:

```text
PDF
DOCX
TXT
MD
```

A private Drive file requiring Google account authentication will not work with this simple public-link approach.

## Important note about persistence

This version keeps processed chunks and embeddings in Streamlit session state.

That means:

- Asking multiple questions does NOT recreate embeddings.
- Adding new documents rebuilds the FAISS index using the complete current chunk list.
- Refreshing/restarting the Streamlit session clears the in-memory index.

For a production application, the next step would be saving the FAISS index and metadata to disk or a persistent vector database.

# Plant Care RAG Chatbot (Bilingual: English + Urdu)

A production-ready Retrieval-Augmented Generation chatbot for plant care guidance,
with bilingual support for English and Urdu queries.

## RAG Enhancement Techniques Used

| Technique | Description |
|-----------|-------------|
| **Cross-lingual Embeddings** | `paraphrase-multilingual-mpnet-base-v2` enables semantic search across Urdu queries against English PDFs |
| **Hybrid Retrieval** | Dense (FAISS) + Sparse (BM25) retrieval combined |
| **Reciprocal Rank Fusion** | Merges dense and sparse ranked lists without score calibration issues |
| **Cross-encoder Reranking** | `ms-marco-MiniLM-L-6-v2` reranks candidates for precision |
| **Multi-turn Context** | Last 6 conversation turns passed to LLM for coherent dialogue |
| **Metadata Enrichment** | Source PDF and page number stored per chunk for citations |
| **Grounding Guard** | System prompt enforces context-only answering to reduce hallucinations |
| **Language Routing** | Automatic Urdu/English detection with script-level heuristics + ML models |

## Setup

```bash
# 1. Enter project directory
cd "PlantCare Chatbot"

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 5. Build the vector index (run once)
python scripts/build_index.py

# 6. Start the API server
uvicorn app.main:app --reload --port 8000
```

## API Usage

### Health Check

`GET http://localhost:8000/api/v1/health`

### Chat (English)

```json
{
  "query": "How often should I water my succulents?",
  "chat_history": []
}
```

### Chat (Urdu)

```json
{
  "query": "میرے پودے کی پتیاں پیلی کیوں ہو رہی ہیں؟",
  "chat_history": []
}
```

### Multi-turn Example

```json
{
  "query": "What fertilizer should I use?",
  "chat_history": [
    {"role": "user", "content": "I have a rose plant"},
    {"role": "assistant", "content": "Rose plants need well-drained soil..."}
  ]
}
```

## Deploy to Render

1. Push your project to a GitHub repository (include the `pdfs/` folder)
2. Create a new Render Web Service connected to your repo
3. Render will auto-detect `render.yaml`
4. Set `OPENAI_API_KEY` as an environment secret in the Render dashboard
5. Deploy — the build step runs `scripts/build_index.py` and persists to a disk mounted at `/app/vectorstore`

## Interactive API Docs

Once running, visit: `http://localhost:8000/docs`


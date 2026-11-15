# Uzhavan RAG Assistant

Chat (left) over Tamil Nadu agriculture knowledge; upload files and scrape the Uzhavan portal (right). Answers come only from indexed data, in Tamil or English, with source links.

## Structure

```
backend/   main.py, config.py, schemas.py, routers/{upload,scrape,chat,documents}.py,
           services/{scraper,excel_writer,parser,chunker,embedder,vector_store,rag,ingest}.py
frontend/  app.py
data/      uploads/, uzhavan_pages.xlsx (generated)
vector_store/   FAISS persistent store (generated)
```

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
playwright install chromium
copy .env.example .env                              # then set LLM_API_KEY
uvicorn backend.main:app --reload
streamlit run frontend/app.py
```

Backend: http://localhost:8000/docs · UI: http://localhost:8501.
The first run downloads `intfloat/multilingual-e5-base` (~1 GB). Set `EMBEDDING_PROVIDER=openai` to use OpenAI embeddings instead (then delete `vector_store/index.faiss` and `metadata.json` and re-index).
`UZHAVAN_BASE_URL` in `.env` sets the site to scrape (default: the TNAgrisnet Uzhavan dashboard); change it if the portal lives elsewhere.

## Pipelines (one FAISS index, same chunker + embedder)

```
A. Upload   file -> parser (PDF/DOCX/TXT/CSV/XLSX) -> clean -> chunker(800/100) -> embedder -> FAISS   [source_type=upload]

B. Web      Playwright (home + internal links, >=10 pages, 3 retries, robots.txt, 1-2 s delay)
              -> data/uzhavan_pages.xlsx -> read back -> chunker(800/100) -> embedder -> FAISS          [source_type=web]

Chat        question -> embed(query) -> top-5 search -> prompt(context only) -> LLM (streamed) -> answer + sources
```

Chunk metadata: `source_type, source_name, url, page, chunk_id` (plus `doc_id`). Re-indexing a file replaces its old chunks; re-scraping replaces all web chunks.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | /upload/file | Upload pipeline |
| POST | /scrape/uzhavan | Web pipeline |
| POST | /chat | `{"question": "...", "stream": false}` → `{answer, sources}`; `stream: true` returns NDJSON |
| GET | /documents | List indexed sources |
| DELETE | /documents/{id} | Remove a source |
| GET | /download/excel | Download scraped Excel |

## Test checklist

1. Start backend and frontend; open the UI.
2. Upload one PDF → "Indexed files" shows it with a chunk count.
3. Click "Scrape Uzhavan website" → success message with pages and chunks.
4. Open the downloaded Excel: ≥10 rows, columns S.No, URL, Title, Headings, Content, Scraped_At.
5. Ask 3 questions: one answerable from the PDF, one from the website, one in Tamil.
6. Check each answer's sources (page URL or `file.pdf (page N)`).
7. Ask an unrelated question (e.g. cricket) → "I could not find this in the indexed data." with no sources.
8. Delete a source and confirm it no longer appears in answers.

# RAG Assignment: Baseline Retrieval vs. Cross-Encoder Re-Ranking

This project compares two retrieval pipelines for question answering over one long PDF (the EU GDPR, Regulation (EU) 2016/679), using exactly the same chunks for both:

1. **Pipeline 1 (baseline):** dense Bi-Encoder embeddings in FAISS, retrieve the Top 10 by cosine similarity, give the Top 3 chunks to the LLM.
2. **Pipeline 2 (re-ranking):** take the same Top 10, score every (query, chunk) pair with a Cross-Encoder, re-sort, give the new Top 3 to the LLM with an identical prompt.

The notebook also stress-tests both pipelines with tricky queries (negation, exceptions, keyword traps, buried chunks), shows rank shifts, compares the LLM answers, and profiles latency and scaling.

## Repository contents

| File | What it is |
|---|---|
| `RAG_Baseline_vs_Reranking.ipynb` | The full notebook (run on Google Colab), with comments explaining each step |
| `document.pdf` | The GDPR PDF used as the corpus |
| `failure_scan_results.csv` | Rank-shift scan over 12 stress queries |
| `latency_table.csv` | Latency of each pipeline stage |
| `outputs/` | Raw notebook outputs from the runs (deep-dive tables and answers) |
| `report.pdf` | The written report |

## Setup used

- **Document:** GDPR, 88 pages
- **Chunking:** 349 chunks, 200 words each, 40-word overlap
- **Bi-Encoder:** `gemini-embedding-001` (768 dimensions, normalised), FAISS `IndexFlatIP` (inner product = cosine similarity)
- **Cross-Encoder:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM:** `gemini-2.5-flash`, temperature 0, same prompt for both pipelines
- **Hardware:** Google Colab CPU

## How to run

1. Open the notebook in Google Colab.
2. Upload `document.pdf` to the Colab Files panel (left sidebar). Keep the name exactly `document.pdf`.
3. Add a Gemini API key (free from https://aistudio.google.com/apikey), either as a Colab Secret named `GEMINI_API_KEY` or by pasting it into the `MANUAL_KEY` variable in the key cell. **Never commit a real key.**
4. Run Runtime > Run all.

Notes:
- The free Gemini tier has rate limits (including a daily request cap), so the notebook paces its requests and retries automatically. Embedding takes a few minutes.
- Setting `EMBED_BACKEND = "local"` in the settings cell switches the embeddings to a local Hugging Face model (`bge-small-en-v1.5`), which is faster and needs no quota.

## What the notebook does

1. Loads and chunks the PDF.
2. Embeds the chunks and builds the FAISS index.
3. Implements both pipelines and runs a demo query.
4. Scans 12 stress queries and flags those where a chunk from baseline rank 4-10 enters the re-ranked Top 3.
5. Runs a deep dive (rank-shift table, top-3 previews, Answer 1 vs Answer 2) for selected queries.
6. Profiles latency per stage and times the Cross-Encoder on growing candidate counts to estimate scaling.

## Key observations

- Re-ranking changed the Top 3 for 11 of 12 stress queries, but this only means the context changed, not that the baseline was wrong.
- It fixed a clear omission for the automated-decision query (the baseline answer missed the exceptions) and helped slightly for the portability query.
- It hurt the erasure query and gave no gain on several others. The Cross-Encoder was trained on web-search data (MS MARCO), not legal text.
- Re-ranking added about 1.8 s per query on a Colab CPU, roughly 12 times the baseline retrieval time. Cross-Encoder cost grows linearly with the number of chunks (about 157 ms per chunk), which is why it is used only on a short candidate list.

See `report.pdf` for the full analysis.

## Limitations

- Small test set (12 queries, a handful of deep dives), with no formal retrieval metrics.
- LLM answers vary slightly between runs even at temperature 0, while retrieval is deterministic.
- Latency was measured on a shared Colab CPU and varies between sessions.
- The Gemini free-tier daily quota limited how many queries could be run with full answers.

## Author

Vadali S S Bharadwaja, Roll No. 2025204012

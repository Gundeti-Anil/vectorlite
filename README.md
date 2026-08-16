# Vector Lite Core

Build a complete, working project called VectorLite — a lightweight, embeddable vector search engine in Python, built from scratch without using faiss, hnswlib, annoy, chromadb, or any other vector-index library for the core indexing logic (numpy is fine for math operations). Requirements: 1. Core Index (flat/brute-force) - A class that stores vectors with associated IDs and metadata - Insert, delete, and top-k similarity search (support both cosine similarity and euclidean distance) - Pure numpy implementation, vectorized (no naive python loops over every vector) 2. Approximate Nearest Neighbor Index (simplified, HNSW-inspired) - A second index class implementing a simplified graph-based ANN search: build a navigable small-world-style graph over inserted vectors, with greedy graph traversal from an entry point to find approximate top-k neighbors - Keep it simplified relative to true HNSW (e.g., single layer or a small number of layers is fine) — prioritize correctness and clarity over full algorithmic fidelity - Expose the same insert/delete/query interface as the flat index so they're interchangeable 3. Persistence Layer - Serialize the full index (vectors, metadata, graph structure) to disk and reload it on startup - Use a sensible format (pickle, numpy .npz, or json+numpy — your choice, but explain the tradeoff in a comment) - Ensure reloading produces an index that behaves identically to the pre-save state 4. Network API (FastAPI) - POST /insert — insert a vector with id + metadata + optional text (accept raw vectors; don't handle embedding generation internally, assume the caller provides embeddings) - POST /query — given a query vector and k, return top-k results with ids, scores, and metadata - DELETE /vectors/{id} — remove a vector - GET /stats — return index size, dimensionality, memory usage - Include a config option to select flat vs. ANN index at startup 5. Benchmarking Script - A separate script that generates synthetic random vectors at increasing scales (e.g., 1k, 10k, 100k vectors) - Measures and plots (matplotlib) query latency and recall (using flat search as ground truth) for the ANN index vs. flat index across these scales - Output results as a table and a saved chart image 6. Project structure - Clean, well-commented, modular file structure (e.g., index/flat.py, index/ann.py, persistence.py, api.py, benchmark.py) - A README explaining architecture, how to run the API, how to run benchmarks, and current limitations - requirements.txt 7. Tests - Basic pytest unit tests: insert/query correctness on the flat index, persistence round-trip correctness, and a sanity check that ANN recall stays above a reasonable threshold (e.g., >85%) on a small synthetic dataset Give me the full project — all files, complete and runnable, not pseudocode or partial snippets. After generating it, give me a short summary of the key design decisions you made and why, so I have a reference to study from afterward.

This project was built with [Lovable](https://lovable.dev).

## Build with Lovable

Continue developing this project in the [Lovable editor](https://lovable.dev/projects/a380e852-84ea-474b-aa6b-f6aeb86f79ce).

- **Ship faster**: describe what you want to build and Lovable handles the code.
- **Stay in sync**: every change made in Lovable is committed straight to this repository.
- **Full ownership**: this code is yours. Push to `main` on GitHub and your changes sync back into Lovable, ready for your next prompt.

## Development

Prefer working locally? You need Node.js and npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

```sh
git clone <this-repository-url>
cd <repository-name>
npm i
npm run dev
```

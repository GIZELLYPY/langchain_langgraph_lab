# langchain_anggraph_lab
A hands-on laboratory for building LLM applications using LangChain and LangGraph. This repo contains experiments, architectural patterns, and practical implementations including RAG pipelines, agent workflows, tool integration, and LLM orchestration.

## Installation

```bash
cd "/home/gizelly/Documents/LLM /langchain_langgraph_lab"
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

This installs the core libraries used in the repo, including `langchain`, `langgraph`, `qdrant-client`, and local embedding dependencies.

## Qdrant Setup

Before running codes that uses Qdrant example

you need a Qdrant server running locally.

### Start Qdrant

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

What this does:

- `6333` exposes Qdrant's HTTP API.
- `6334` exposes Qdrant's gRPC API.
- `qdrant/qdrant` is the official Docker image.

### Verify That Qdrant Is Running

```bash
curl http://localhost:6333
curl http://127.0.0.1:6333
```

Both commands should return JSON similar to:

```json
{"title":"qdrant - vector search engine","version":"..."}
```

If you get `Connection refused`, Qdrant is not running or the port is not exposed correctly.

### Useful Qdrant Endpoints

```bash
curl http://127.0.0.1:6333/collections
curl http://127.0.0.1:6333/collections/rag_articles
```

What they are for:

- `/collections` lists all collections stored in Qdrant.
- `/collections/rag_articles` shows details about the `rag_articles` collection used by the example script.

The example in [experiments/vector_store_qdrant.py](/home/gizelly/Documents/LLM%20/langchain_langgraph_lab/experiments/vector_store_qdrant.py#L100) stores PDF chunks in the `rag_articles` collection.

### Dashboard

Open this in your browser:

```text
http://127.0.0.1:6333/dashboard
```

If your Qdrant build exposes the dashboard, you can inspect collections and data more easily there than through raw JSON.

### Qdrant vs FAISS

Use FAISS when:

- you want a simple local vector index
- you are experimenting on one machine
- you do not need a database server or HTTP API

Use Qdrant when:

- you want a vector database, not only an index library
- you need collections and persistent storage
- you want metadata filtering
- you want other applications to query vectors through an API

Short version:

- FAISS is lighter and simpler for local experiments.
- Qdrant is more suitable when the project starts to look like a real application or service.

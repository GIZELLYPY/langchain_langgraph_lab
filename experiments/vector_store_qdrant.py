"""Vectors databases are one of those stange things... instead of store rows or documents,
they stores points in a high-dimensional space. The embbedings - those 384-dimensional 
vectors on the code experiments/embbeding_free.py become coordinates. 
The Retrieval IS GEOMETRY!

A vector database indexes and stores vector embeddings for fast retrieval and similarity search.
Vectors database:
- Pinecone, Qdrant, Weaviate, Milvus, Chroma, Redis, FAISS

- FAIS: Facebook AI Similarity Search is a library for fast similarity search of vectors. This runs locally in memory and 
        often used for experiments, prototypes and research. It is not a full database not built-in metadata filtering,
        replication or distribuited scalin.
        Typical use:
            Small to medium datasets
            Local RAG experiments
            Academic Projects

- Pinecone: Fully managed vector database service designed for production AI applications. It's cloude native - 
            vector database as a service -  handles scaling,
            indexing and infrastructure, supports metadata filtering, often used in enterprise RAG systems.
            Typical use:
                Production AI applications
                SaaS AI products
                Systems needing high scalability


- Qdrant: Is an open-source vector database optimized for semanct search and AI applications, modern vector
          database with string performance and filtering. It's writen in rusty, very fast and memory efficente, 
          supports vector search + metadata filtering, can run locally or in the cloud. 
            Typical use:
                RAG pipelines
                AI agents
                recommendation systems

                
- Weaviate: Is an open-source vector database with built-in AI modules. Features: vector dearch, 
            hybrid search (vector + keyword/BM25), GraphQL API, Built-in embeddingt modules.
            Typical use:
                knowledge search systems
                AI knowledge graphs
                enterprise semantic search

- Milvus: Is a high-performance distribuited vector database designed for large-scale AI workloads. It's handles billions of vectors,
          GPU acceleration, Distributed architecture, often used in large ml systems.
          Typical use:
                Computer vision search
                Large AI platforms
                Enterprise AI infrastructure

- Chroma: Is a simple open-source vector databse designed for LLM applications. It's very easy to use, Python-friendly, 
          often used with LangChain and also considered a beginner-friendly vector database for LLM projects.
          Typical use:
                RAG prototypes
                LLM experimentation
                small AI apps

- Redis: A real-time database that can also stores vectors. Is an in-memory databse that now supports vector search through Redis Vector Similarity Search.
         It's extremelly fast, supports vector + key-value+caching, often used for real-time AI applications.
         Typical use:
                AI recomendations systems
                semantic caching
                low-latency search

"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings

from qdrant_client import QdrantClient


from dotenv import load_dotenv

load_dotenv(".venv/env")

loader = PyPDFLoader("./data/artigo.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = splitter.split_documents(docs)

# Embedding model from huggingFace
# Why use HuggingFaceEmbeddings here:
# If the rest of the code uses LangChain vector stores, retrievers, or indexing APIs, 
# they expect an embeddings object with methods like embed_query() and embed_documents().
# SentenceTransformer("all-MiniLM-L6-v2") (from sentence_transformers import SentenceTransformer) alone 
# does not match that LangChain interface.
# HuggingFaceEmbeddings internally uses a sentence-transformers model, 
# but exposes it in the format LangChain expects.
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# collection_name: That means all embedded chunks from 
#  PDF are stored in a Qdrant collection called my_pdf_article
vector_store = Qdrant.from_documents(
    chunks,
    embedding_model,
    url="http://localhost:6333",
    collection_name="my_pdf_article"
)

print("Documents indexed in Qdrant")

query = "What is retrieval augmented generation?"

results = vector_store.similarity_search(query)

for r in results:
    print(r.page_content)
"""Route a question to the right Qdrant collection and retrieve docs.

This script:
1. Loads a small set of official Python and JavaScript docs pages
2. Splits and stores them in two Qdrant collections
3. Routes each user question with a free Hugging Face model
4. Retrieves the most relevant chunks from the selected collection
"""

from typing import Literal

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from transformers import pipeline


QDRANT_URL = "http://127.0.0.1:6333"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

PYTHON_COLLECTION = "python_official_docs"
JS_COLLECTION = "javascript_official_docs"

PYTHON_DOC_URLS = [
    "https://docs.python.org/3/tutorial/index.html",
    "https://docs.python.org/3/tutorial/stdlib.html",
    "https://docs.python.org/3/tutorial/datastructures.html",
    "https://docs.python.org/3/library/functions.html",
]

JS_DOC_URLS = [
    "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide",
    "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise",
    "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/async_function",
    "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array",
]


class RouteQuery(BaseModel):
    """Validated route chosen for a user query."""

    datasource: Literal["python_docs", "js_docs"] = Field(
        ...,
        description="Choose the most relevant datasource for the user's question.",
    )


LABEL_TO_ROUTE = {
    "Python documentation and Python language questions": "python_docs",
    "JavaScript documentation and JavaScript language questions": "js_docs",
}


def build_router():
    """In this setup, facebook/bart-large-mnli is being used only as a classifier.
    Example
    {
    "sequence": "How do JavaScript promises work?",
    "labels": [
        "JavaScript documentation and JavaScript language questions",
        "Python documentation and Python language questions"
    ],
    "scores": [0.98, 0.02]
    }


    """
    return pipeline(
        task="zero-shot-classification",
        model="facebook/bart-large-mnli",
    )


def build_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def load_and_split(urls: list[str], source_name: str):
    docs = WebBaseLoader(web_paths=tuple(urls)).load()

    for doc in docs:
        doc.metadata["source_group"] = source_name

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
    )
    return splitter.split_documents(docs)


def index_collection(
    urls: list[str],
    collection_name: str,
    source_name: str,
    embeddings: HuggingFaceEmbeddings,
) -> Qdrant:
    
    """This function loads documents from a 
    list of URLs, splits them into chunks, embeds those chunks, and stores 
    them in a Qdrant collection"""
    chunks = load_and_split(urls, source_name)

    return Qdrant.from_documents(
        chunks,
        embeddings,
        url=QDRANT_URL,
        collection_name=collection_name,
        force_recreate=True,
    )


def build_doc_stores() -> dict[str, Qdrant]:
    embeddings = build_embeddings()

    python_store = index_collection(
        urls=PYTHON_DOC_URLS,
        collection_name=PYTHON_COLLECTION,
        source_name="python_docs",
        embeddings=embeddings,
    )
    js_store = index_collection(
        urls=JS_DOC_URLS,
        collection_name=JS_COLLECTION,
        source_name="js_docs",
        embeddings=embeddings,
    )

    return {
        "python_docs": python_store,
        "js_docs": js_store,
    }


def route_query(question: str) -> RouteQuery:
    router = build_router()

    result = router(
        sequences=question,
        candidate_labels=list(LABEL_TO_ROUTE.keys()),
        hypothesis_template="This question is about {}.",
    )

    best_label = result["labels"][0]
    return RouteQuery(datasource=LABEL_TO_ROUTE[best_label])


def retrieve_from_route(question: str, stores: dict[str, Qdrant]):

    # This runs the classifier first. It returns something like RouteQuery(datasource="python_docs")
    route = route_query(question)

    # route.datasource == "python_docs", this becomes store = stores["python_docs"]
    # "python_docs" -> Python Qdrant vector store
    # "js_docs" -> JavaScript Qdrant vector store
    store = stores[route.datasource]


    # That means the classifier does not directly talk to Qdrant.
    # It only chooses a key. Then the Python code uses that key to pick the correct Qdrant store.
    retriever = store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20},
    )
    docs = retriever.invoke(question)

    return route, docs


def print_results(question: str, route: RouteQuery, docs) -> None:
    collection_name = (
        PYTHON_COLLECTION if route.datasource == "python_docs" else JS_COLLECTION
    )

    print(f"Question: {question}")
    print(f"Route: {route.datasource}")
    print(f"Qdrant collection: {collection_name}")
    print("Retrieved chunks:")

    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        print(f"\n--- DOC {index} ---")
        print(f"Source: {source}")
        print(doc.page_content[:400])

    print("\n" + "=" * 60)


def main() -> None:
    stores = build_doc_stores()

    questions = [
        "How do I create a virtual environment in Python?",
        "How do JavaScript promises work?",
        "What does the Python standard library provide?",
        "What is async function in JavaScript?",
    ]

    for question in questions:
        route, docs = retrieve_from_route(question, stores)
        print_results(question, route, docs)


if __name__ == "__main__":
    main()

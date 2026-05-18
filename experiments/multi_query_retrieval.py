"""Multi-query retrieval with Qdrant + a local Hugging Face model.

Flow:
Web page -> chunks -> embeddings -> Qdrant -> multiple retrieval queries
-> merged context -> prompt -> answer
"""

from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


load_dotenv(".venv/env")


SOURCE_URL = "https://en.wikipedia.org/wiki/Ancient_Greek_philosophy"
COLLECTION_NAME = "wikipedia_ancient_greek_philosophy_multi_query"
QUESTION = "Who are some key figures in the ancient Greek history of philosophy?"


def build_vector_store() -> Qdrant:
    """Loads the page, chunks it, embeds it, and stores it in Qdrant."""
    loader = WebBaseLoader(SOURCE_URL)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(docs)

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return Qdrant.from_documents(
        chunks,
        embedding_model,
        url="http://127.0.0.1:6333",
        collection_name=COLLECTION_NAME,
    )


def generate_query_variants(question: str) -> list[str]:
    """Creates multiple query rewrites."""
    return [
        question,
        "Important philosophers in ancient Greek philosophy",
        "Main thinkers in the history of ancient Greek philosophy",
        "Ancient Greek philosophy major figures and schools",
    ]


def retrieve_context(vector_store: Qdrant, question: str) -> str:
    """Runs retrieval for each query and deduplicates the returned chunks."""
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20},
    )

    unique_chunks: dict[str, str] = {}
    for query in generate_query_variants(question):
        docs = retriever.invoke(query)
        for doc in docs:
            content = doc.page_content.strip()
            if content and content not in unique_chunks:
                unique_chunks[content] = query

    return "\n\n".join(unique_chunks.keys())


def answer_question(context: str, question: str) -> str:
    """sends the merged context to flan-t5-base."""
    prompt = ChatPromptTemplate.from_template(
        """Answer the question using only the context below.
Do not copy a section title as the answer.
Write 2-4 clear sentences.
If the answer is not in the context, say so.

Context:
{context}

Question: {question}

Answer:"""
    )

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    prompt_text = prompt.invoke({"context": context, "question": question}).to_string()

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )
    outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main() -> None:
    vector_store = build_vector_store()
    context = retrieve_context(vector_store, QUESTION)
    response = answer_question(context, QUESTION)

    print("retrieved context chunks -----------")
    print(context[:1500])
    print("-----------")
    print("########################")
    print(response)
    print("########################")


if __name__ == "__main__":
    main()

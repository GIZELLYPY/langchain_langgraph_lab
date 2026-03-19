"""Web page -> Chunking -> Embeddings -> Qdrant Vector Database -> Retrieve relevant text -> Prompt + LLM -> Answer
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from qdrant_client import QdrantClient
from langchain_community.document_loaders import WebBaseLoader

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from dotenv import load_dotenv

load_dotenv(".venv/env")

loader = WebBaseLoader("https://en.wikipedia.org/wiki/Ancient_Greek_philosophy")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=100,
)

chunks = splitter.split_documents(docs)


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Qdrant.from_documents(
    chunks,
    embedding_model,
    url="http://127.0.0.1:6333",
    collection_name="wikipedia_ancient_greek_philosophy",
)


prompt = ChatPromptTemplate.from_template(
    """Answer the question using only the context below.
Do not copy a section title as the answer.
Write 1-2 clear sentences.
If the answer is not in the context, say so."

Context:
{context}

Question: {question}

Answer:"""
)


tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

question = "Who are the key figures in the ancient greek history of philosophy?"

# Retriever flow
# This converts the vector store into a retriever object.
# fetch_k First get X relevant candidate chunks from the vector store.
# K Then choose the best X from those fetch_k .


retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k":50}
)



# retriever is a simpler interface whose job is: given a question, return relevant documents
# embeds the question
# searches Qdrant for similar vectors
# returns the most relevant Document chunks
docs = retriever.invoke(question) 

print("retrieved context chunks -----------")
for i, doc in enumerate(docs, 1):
    print(f"\n--- DOC {i} ---\n")
    print(doc.page_content[:300])
print("-----------")


# This extracts only the text from each document and combines them into one string.
context = "\n\n".join(doc.page_content for doc in docs)

# Build the final prompt the LLM will see
prompt_text = prompt.invoke({"context": context, "question": question}).to_string()

# input, user
# Hugging Face models expect tensor inputs
# pt means PyTorch
# return_tensors="pt" Don’t return plain Python lists. Return PyTorch tensors.
inputs = tokenizer(
    prompt_text,
    return_tensors="pt", 
    truncation=True,
    max_length=2048,
)
outputs = model.generate(**inputs, max_new_tokens=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)


print("########################")
print(response)
print("########################")

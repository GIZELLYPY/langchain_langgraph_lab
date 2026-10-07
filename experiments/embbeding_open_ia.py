from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv(".venv/env")

"""
    vector_model = OpenAIEmbeddings()

it may look like a local library doing some clever math on your machine. It isn’t. The computation happens on OpenAI’s 
servers. OpenAIEmbeddings is simply a client that sends your text to an 
external API where a large neural network converts that text into vectors (embeddings). Those vectors are then 
returned to your program and used for retrieval systems, semantic search, or RAG pipelines.

Because the embedding model runs on remote infrastructure — GPUs, distributed systems, storage, networking — 
the API is a paid service. That means an OpenAI account with active billing and available credit is required. 
If the account has no credit, 
the API refuses the request and returns an error like: 429 RateLimitError: insufficient_quota

    """


loader = PyPDFLoader("./data/artigo.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                          chunk_overlap=200,
                                          )

chunks = splitter.split_documents(docs)

vector_model = OpenAIEmbeddings()


each_chunk = [chunk.page_content for chunk in chunks]

embeddings = vector_model.embed_documents(each_chunk)

print(type(embeddings))
print(len(embeddings))
print(len(embeddings[0]))
print(embeddings[0][:10])

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


from dotenv import load_dotenv

load_dotenv(".venv/env")

"""
There is a pleasant little trick in the LLM ecosystem: not everything needs a paid API. 
The code is using one of the classic escape hatches — local embeddings. No remote GPUs. 
No billing meter quietly ticking. Just mathematics running on your own machine.
why it works without OpenAI credits?

SentenceTransformer downloads a pretrained model from Hugging Face Hub and runs it locally. 
The model all-MiniLM-L6-v2 is a compact transformer trained specifically for generating sentence embeddings.

Each chunk becomes a numeric vector (typically 384 dimensions for this model). 
Those vectors capture semantic similarity. 
Two texts about the same concept will end up close together in vector space.

There is a warning: 
        Warning: You are sending unauthenticated requests to the HF Hub. 
        Please set a HF_TOKEN to enable higher rate limits and faster downloads.

This is not an error. 
The model downloaded successfully. Hugging Face simply warns that anonymous downloads have lower rate limits.

If you want to remove the warning and get faster downloads, create a free Hugging Face token.

Create account
https://huggingface.co

Generate token
https://huggingface.co/settings/tokens
    """

loader = PyPDFLoader("./data/artigo.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = splitter.split_documents(docs)

each_chunk = [chunk.page_content for chunk in chunks]


model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(each_chunk)

print(embeddings)

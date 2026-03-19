from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("./data/artigo.pdf")
docs = loader.load()

# creates a splitter from LangChain, it splits text into chunks
# chunk_size = 1000 -> each chunks should have about 1000 characteres
# chunk_overlap=200 -> the next chunk repeats 200 characteres from the previous chunk
# Why overlaps exists? It preserves context between chunks. Useful for embeddins, retrieval and question answering
# prevents important text from being cut exactly at boundaries.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = splitter.split_documents(docs)

# splitter = the tool
# chunks = the result after splitting

print(type(chunks))
print(len(chunks))
print(chunks[0])
print(chunks[0].page_content[:500])
print(chunks[0].metadata)

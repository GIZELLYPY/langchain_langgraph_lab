from langchain_community.document_loaders import TextLoader

loader = TextLoader("./data/data.txt")
docs = loader.load()

print("type: ", type(docs))
print("len: ", len(docs))
print("docs[0]: ", docs[0])
print(
    "docs[0] is the full Document "
    "object representation, and that representation already includes page_content and metadata"
)
print("docs[0].page_content ", docs[0].page_content)
print("docs[0].metadata): ", docs[0].metadata)

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

PYTHON_CODE = """
def hello_world():
    return "Hello world"

#call the function
hello_world()
"""

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=50, chunk_overlap=10
)

chunks = python_splitter.create_documents([PYTHON_CODE])

print(type(chunks))
print(len(chunks))
print(chunks[0])
print(chunks[0].page_content[:500])
print(chunks[0].metadata)

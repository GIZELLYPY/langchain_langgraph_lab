from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

MARKDOWN_TEXT = """
## Subtitle Section
This is a simple markdown file for testing **bold text**.
#### Final Note

"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN, chunk_size=60, chunk_overlap=40
)

# Here in source its added a simple metadata so is easy to identify were each chunk comes from
chunks = splitter.create_documents([MARKDOWN_TEXT], [{"source": "Atlas Notebook"}])

print(type(chunks))
print(len(chunks))
print(chunks[0])
print(chunks[0].page_content[:500])
print(chunks[0].metadata)

from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """ """

splitter = RecursiveCharacterTextSplitter(
    chunk_size= 100,
    chunk_overlap =0,
)

print(splitter.split_text(text))
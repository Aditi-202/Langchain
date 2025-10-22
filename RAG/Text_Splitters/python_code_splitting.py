from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

text = """ """

splitter = RecursiveCharacterTextSplitter(
    language = Language.py,
    chunk_size= 100,
    chunk_overlap =0,
    separator=''
)

print(splitter.split_text(text))
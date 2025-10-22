from langchain.text_splitter import CharacterTextSplitter

text = """ """

splitter = CharacterTextSplitter(
    chunk_size= 100,
    chunk_overlap =0,
    separator=''
)

print(splitter.split_text(text))
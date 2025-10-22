from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path= 'folder_test',
    glob='*.pdf', # tells you the pattern of loading the pdf from folder 
    loader_cls=PyPDFLoader
)

docs = loader.load()
print(docs)



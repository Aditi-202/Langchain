from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('Taylor_Scientific_Management_Revision.pdf')
docs = loader.loader.load()

print(docs)
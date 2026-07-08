from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('smartdoc_publish.pdf')

docs = loader.load()

print(docs[0].metadata)
print(docs[0].page_content)

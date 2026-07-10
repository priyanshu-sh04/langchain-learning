from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Your socurce documents 
document = [
    Document(page_content="Langchian helps developers to build LLM application easily."),
    Document(page_content="Chroma is a vector database optimised for LLM-based search."),   
    Document(page_content="Emebeddings convert tecxt into high-dimensional vectors."),
    Document(page_content="OpenAi provides powerful embedding models")
]

# Intialise embedding model
embedding_model = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

# Create Chroma vector store in memory
vectore_store = Chroma.from_documents(
    documents=document,
    embedding=embedding_model,
    collection_name="my_collection"
)

retriever = vectore_store.as_retriever(search_kwargs={"k":2})

query = "What is Chorma used for?"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n--- {i + 1} ---")
    print(doc.page_content)
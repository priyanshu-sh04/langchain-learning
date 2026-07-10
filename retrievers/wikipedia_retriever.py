from langchain_community.retrievers import WikipediaRetriever

# Intialise the retriever
retriver = WikipediaRetriever(
    top_k_results=2,
    lang="en"
)

# Define your query
query = "India Pakistan geopolitical history China perspective"

# Get relevant Wikipedia documents
docs = retriver.invoke(query)
print(docs)

# Print retrieved content
for i, doc in enumerate(docs):
    print(f"\n--- {i + 1} ---")
    print(f"content:\n{doc.page_content}") # truncate the display
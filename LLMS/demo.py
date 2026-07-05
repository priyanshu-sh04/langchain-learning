from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")

result = llm.invoke('What is AI , Hindi mein samjhao')

print(result.content)

# langchain adviced not to use llms they are to old 
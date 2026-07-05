from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage 
from dotenv import load_dotenv

load_dotenv()

chat_model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=1.5,
    max_tokens=60
    )

messages = [
    SystemMessage('You are a helpful assistant kai who explains things simply.'),
    HumanMessage('what is Ai Agents')
]

result = chat_model.invoke(messages)
print(result.content, end="\n\n")
print(result.response_metadata) # token usages btayega
4
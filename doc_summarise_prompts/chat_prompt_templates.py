from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

chat_template = ChatPromptTemplate([
    ('system', 'you are a helpful {domain} expert'),
    ('human', 'Explain in simple terms, what is {topic}'),

    """ 
    bcz of some weired behaviour of langchain this will not get us what we want 
    
    SystemMessage(content='you are a helpful {domain} expert'),
    HumanMessage(content='Explain in simple terms, what is {topic}')"""
])

prompt = chat_template.invoke({'domain':'cricket', 'topic':'virat kohli'})

print(prompt)
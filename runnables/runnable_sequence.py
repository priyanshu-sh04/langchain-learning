from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

prompt = PromptTemplate(
    template="Write a joke on the topic {topic}",
    input_variables=['topic']
)

model = ChatGroq(model="llama-3.1-8b-instant")

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template="Explain the following joke {text}",
    input_variables=['text']
)

chain = RunnableSequence(prompt | model | parser | prompt2 | model | parser)

result = chain.invoke({'topic':'AI'})

print(result)
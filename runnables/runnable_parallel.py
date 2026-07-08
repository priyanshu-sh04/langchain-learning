from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template="Genrate a tweet on topic {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Genrate a linkdin post on topic {topic}",
    input_variables=['topic']
)

model = ChatGroq(model="llama-3.1-8b-instant")

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, model, parser),
    'linkdin': RunnableSequence(prompt2, model, parser)
})

result = parallel_chain.invoke({'topic':'AI'})

print(result['linkdin'])
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template="Genrate a report on the topic {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="summarize this report {text}",
    input_variables=['text']
)

model = ChatGroq(model="llama-3.1-8b-instant")

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    (lambda x : len(x.split())>100, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(joke_gen_chain, branch_chain)
print(final_chain.invoke({'topic':'America Army vs Russian Army'}))
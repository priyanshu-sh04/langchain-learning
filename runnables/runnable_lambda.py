from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

prompt = PromptTemplate(
    template="Write a joke on the topic {topic}",
    input_variables=['topic']
)

def wordcounter(text):
    return len(text.split())

runnable_word_counter = RunnableLambda(wordcounter)

model = ChatGroq(model="llama-3.1-8b-instant")

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableSequence(prompt, model, parser, runnable_word_counter)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

print(final_chain.invoke({'topic':'ethical hacking'}))

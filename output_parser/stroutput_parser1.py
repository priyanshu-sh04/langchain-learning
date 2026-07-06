from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="task-generation"
)

model = ChatHuggingFace(llm=llm)

template1 =PromptTemplate(
    template='Write a detailed report on {topic}',
    input_types=['topic']
)

template2 =PromptTemplate(
    template='Write a 10 line summary on the following text ./n {text}',
    input_types=['text']
)


parser = StrOutputParser()

chain = template1 | model | parser | template2 | model

result = chain.invoke({'topic':'Black-Hole'})

print(result)
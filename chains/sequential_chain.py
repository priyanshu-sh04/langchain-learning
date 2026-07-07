from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="task-generation"
)

prompt1 = PromptTemplate(
    template="Genrate a detail report on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Genrate a 5 line summary from the following text \n {summary}",
    input_variables=['summary']
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic':'unemployment in India'})

print(result)

chain.get_graph().print_ascii()
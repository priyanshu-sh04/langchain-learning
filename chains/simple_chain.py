from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="task-generation"
)

prompt = PromptTemplate(
    template="Write five line summary about {topic}",
    input_variables=['topic']
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'topic':'DSA'})

print(result)

chain.get_graph().print_ascii()
from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup
from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant")

prompt = PromptTemplate(
    template='Write a Answer the the following question \n {question} from the following text -\n {text}',
    input_variables=['question', 'text']
)

parser = StrOutputParser()

url ='https://www.google.com/search?q=attention+all+you+need&sxsrf=APpeQnutw58iLBT0MP9pAGdKC8u1PSixDQ%3A1783524488270&gs_ssp=eJzj4tVP1zc0TEkrSU7JzjE3YPQSSywpSc0ryczPU0jMyVGozC9VyEtNTQEA_jwNSw'
loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | model | parser
result = chain.invoke({'question':'What is attention all you need and when it introduce', 'text':docs[0].page_content})
print(result)

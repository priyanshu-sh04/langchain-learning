from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
from dotenv import load_dotenv
import time 

load_dotenv()

set_llm_cache(InMemoryCache())

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",  
)

model = ChatHuggingFace(llm=llm)

start = time.time()
result1 = model.invoke("what is the capital of US ?")
print(f"1st call: {time.time() - start:.2f}s")
print(result1.content)

start = time.time()
result2 = model.invoke("Tell me something special about capital of US !!")
print(f"2nd call : {time.time() - start:.2f}s")
print(result2.content)


# for chunk in model.stream("Tell me a short story about a robot"):
   # print(chunk.content, end="", flush=True)    
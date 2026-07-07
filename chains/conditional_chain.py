from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant")


parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive','negative'] = Field(
        description='Give the sentiment of the feedback'
    )

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt = PromptTemplate(
    template="Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}",
    input_variables=['feedback'],
    partial_variables= {'format_instruction': parser2.get_format_instructions()}
)


classifier_chain = prompt | model | parser2

prompt1 = PromptTemplate(
    template="Write an approprite resopnse to this positive feedback \n {feedback} "
)

prompt2 = PromptTemplate(
    template="Write an approprite resopnse to this negative feedback \n {feedback} "
)


branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt1 | model | parser ),
    (lambda x:x.sentiment == 'negative', prompt2 | model | parser ),
    RunnableLambda(lambda x: "could not find sentiment")
)

final_chain = classifier_chain | branch_chain

print(final_chain.invoke({'feedback':'This is a beautiful phone'}))

final_chain.get_graph().print_ascii()





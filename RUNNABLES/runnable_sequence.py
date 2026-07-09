from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

model= ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
)

prompt=PromptTemplate(
    template="give me 5 interesting fact about following : {topic}",
    input_variables=["topic"]
)

parser=StrOutputParser()

chain = RunnableSequence(prompt ,model , parser)

print(chain.invoke({"topic":"mahabharata"}))
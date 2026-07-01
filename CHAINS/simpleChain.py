from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt=PromptTemplate(
    template="You are a helpful assistant. give me five interesting facts about {question}",
    input_variables=["question"]
)
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7
)
parser=StrOutputParser()

chain=prompt|model|parser

result=chain.invoke({"question":"tehri garhwal"})

print(result)

chain.get_graph().print_ascii()
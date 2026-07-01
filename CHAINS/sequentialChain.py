from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1=PromptTemplate(
    template="You are a helpful assistant. give me detailed report on {topic}",
    input_variables=["topic"]
)

prompt2=PromptTemplate(
    template="You are a helpful assistant. give me five interesting facts about {report}",
    input_variables=["report"]
)

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

parser=StrOutputParser()

chain=prompt1|model|parser|prompt2|model|parser

result=chain.invoke({"topic":"btech as a professional course"})
print(result)

chain.get_graph().print_ascii()
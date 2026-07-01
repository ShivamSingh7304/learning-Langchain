from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel



with open(r"C:\Users\Shivam\OneDrive\Desktop\LANGCHAIN\CHAINS\example.txt", "r", encoding="utf-8") as file:
    text = file.read()



load_dotenv()

model1 = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
)

model2 = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7,
)

prompt1 = PromptTemplate(
    template="You are a helpful assistant. give me 100 word notes from following text: {text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="You are a helpful assistant.give me 5 quizzes from following  : {text}",
    input_variables=["text"]
)

prompt3=PromptTemplate(
    template="merge following notes and quizzes into a single document:notes-> {notes} and quizzes-> {quizzes}",
    input_variables=["notes", "quizzes"]
)

parser=StrOutputParser()

parallel_chain = RunnableParallel(
    notes=prompt1 | model1 | parser,
    quizzes=prompt2 | model2 | parser
)
merge_chain=prompt3|model1|parser

chain=parallel_chain|merge_chain


result=chain.invoke({"text":text}) 

print(result)

chain.get_graph().print_ascii()
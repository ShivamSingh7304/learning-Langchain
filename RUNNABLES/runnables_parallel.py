from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence
from dotenv import load_dotenv
 
load_dotenv()

model =ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
)

prompt1 = PromptTemplate(
    template="You are a helpful assistant. give me 100 word notes from following topic: {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="You are a helpful assistant.give me 5 quizzes from following  : {topic}",
    input_variables=["topic"]
)

prompt3=PromptTemplate(
    template="merge following notes and quizzes into a single document:notes-> {notes} and quizzes-> {quizzes}",
    input_variables=["notes", "quizzes"]
)

parser=StrOutputParser()

parallel_chain = RunnableParallel(
    notes=RunnableSequence(prompt1 ,model , parser),
    quizzes=RunnableSequence(prompt2 ,model , parser)
)
merge_chain=RunnableSequence(prompt3 ,model , parser)

chain=parallel_chain|merge_chain


result=chain.invoke({"topic":"mahabharata"}) 

print(result)

chain.get_graph().print_ascii()


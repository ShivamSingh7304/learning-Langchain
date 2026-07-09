from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence, RunnableParallel 
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

prompt2=PromptTemplate(
    template="give me 3 key highlights from following 5 facts : {facts}",    
    input_variables=["facts"]
)   

parser=StrOutputParser()

fact_chain=RunnableSequence(prompt ,model , parser) 

parallel_chain=RunnableParallel(
    facts=RunnablePassthrough(),
    highlights=RunnableSequence(prompt2 ,model , parser)
)

chain=fact_chain|parallel_chain

result=chain.invoke({"topic":"mahabharata"})

print(result)

print(chain.get_graph().print_ascii())
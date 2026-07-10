from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

#loading docs 
loader= TextLoader('EXAMPLE_FILES/football.txt', encoding='utf-8')

docs = loader.load()

'''
print(docs)
print(type(docs))
print(len(docs))
print(docs[0].page_content)
print(type(docs[0]))
'''

model= ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
)

prompt= PromptTemplate(
    template='Summarize the following text in 100 words \n {docs}', 
    input_variables=['docs']
)   

parser= StrOutputParser()


chain = prompt | model | parser 

result = chain.invoke({'docs': docs[0].page_content})

print(result)
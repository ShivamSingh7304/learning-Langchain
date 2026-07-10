#for suppressing the deprecation warning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv

load_dotenv()


from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser


#loading docs
url = "https://www.geeksforgeeks.org/artificial-intelligence/introduction-to-langchain/"

loader = WebBaseLoader(url)
docs = loader.load()

'''
print(len(docs))  #gives the number of pages in the web page
print(docs[0].page_content)  #printing the content of the first page of the web page
'''

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
)

prompt = PromptTemplate(
    template='ans the following que:{que} \n from text :{text}', 
    input_variables=['que','text']    
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'que': 'highlight some important points from text','text': docs[0].page_content})

print(result)

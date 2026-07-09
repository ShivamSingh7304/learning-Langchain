#for suppressing the deprecation warning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("DOCUMENT_LOADERS/NIPS-2017-attention-is-all-you-need-Paper.pdf")
docs = loader.load()

#print(docs)  #printing the docs object which is a list of Document objects

print(len(docs))   #gives the number of pages in the pdf

print(docs[1].page_content)  #printing the content of the second page of the pdf
# Getting Started with ChromaDB
 
## What it is
 
ChromaDB is a **vector database** — it stores embeddings (text converted to numbers) and lets you search them by meaning. It's the easiest vector store to start with in LangChain: no server setup, works locally out of the box.
 
## Install
 
```bash
pip install langchain-chroma
```
 
## Minimal working example
 
```python
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
 
# 1. Pick an embedding model (converts text → vectors)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
 
# 2. Create the store from your chunked documents
vectorstore = Chroma.from_documents(
    documents=chunks,              # list of Document objects (after splitting)
    embedding=embeddings,
    persist_directory="./chroma_db",   # saves to disk
)
 
# 3. Search it
results = vectorstore.similarity_search("What is CBT?", k=3)
for doc in results:
    print(doc.page_content)
```
 
## Loading it back later (without re-embedding)
 
```python
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)
```
 
Because `persist_directory` was set, Chroma already saved everything to disk — you don't need to redo the embedding step every time you run your app.
 
## Turning it into a Retriever (for LCEL chains)
 
```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
retriever.invoke("What is CBT?")
```
 
This is what plugs into your `{"context": retriever, "question": RunnablePassthrough()}` chain pattern.
 
## Adding more documents later
 
```python
vectorstore.add_documents(new_chunks)
```
 
No need to rebuild from scratch.
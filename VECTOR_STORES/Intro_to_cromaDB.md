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
---

# Vector Store Basic Operations — Beginner Notes
 
Once your Chroma (or FAISS) store is built, here are the core operations you'll use day-to-day.
 
---
 
## 1. `similarity_search` — the basic search
 
Returns the top `k` most similar `Document` objects. No scores, just the documents themselves.
 
```python
results = vectorstore.similarity_search("What is CBT?", k=3)
 
for doc in results:
    print(doc.page_content)
    print(doc.metadata)
```
 
**When to use:** you just want the relevant text, don't care exactly *how* similar each result is.
 
---
 
## 2. `similarity_search_with_score` — search + a number telling you how close the match is
 
Same as above, but also returns a **distance score** for each result — how far apart the query and the chunk are in vector-space.
 
```python
results = vectorstore.similarity_search_with_score("What is CBT?", k=3)
 
for doc, score in results:
    print(f"Score: {score:.4f}")
    print(doc.page_content)
    print("---")
```
 
**Important gotcha:** in Chroma, this returns a **distance**, not a similarity — so **lower = more similar** (0 would mean an exact match). This trips people up because it's the opposite of what "score" usually implies.
 
```python
# Lower distance = better match
sorted_results = sorted(results, key=lambda x: x[1])
```
 
**When to use:** you want to filter out weak matches — e.g. "only keep results with distance below 0.5" — instead of blindly trusting the top `k`.
 
---
 
## 3. `similarity_search_by_vector` — search using a vector you already have
 
Sometimes you've already embedded something yourself and don't want to re-embed it.
 
```python
query_vector = embeddings.embed_query("What is CBT?")
results = vectorstore.similarity_search_by_vector(query_vector, k=3)
```
 
**When to use:** rare in normal RAG flow, but useful if you're caching embeddings or comparing a document's vector against the store (not just a text query).
 
---
 
## 4. `max_marginal_relevance_search` (MMR) — diverse results, not just closest
 
Avoids returning several near-duplicate chunks by balancing relevance with diversity.
 
```python
results = vectorstore.max_marginal_relevance_search(
    "What is CBT?",
    k=3,          # final results returned
    fetch_k=10,   # candidate pool to pick the diverse k from
)
```
 
---
 
## 5. `add_documents` — add new chunks without rebuilding
 
```python
vectorstore.add_documents(new_chunks)
```
 
Useful when new source files come in — you don't need to re-embed everything from scratch, just add the new pieces.
 
---
 
## 6. `delete` — remove documents
 
```python
vectorstore.delete(ids=["doc_id_1", "doc_id_2"])
```
 
Every stored chunk has an ID (auto-generated unless you assign your own). Useful for removing outdated or incorrect source material.
 
---
 
## 7. Filtering by metadata
 
Narrow the search to a subset of your documents before comparing vectors.
 
```python
results = vectorstore.similarity_search(
    "anxiety coping strategies",
    k=3,
    filter={"source": "cbt_handbook.pdf"},
)
```
 
**When to use:** you have multiple categories of content (e.g. CBT vs. mindfulness vs. crisis resources) and want to search only within one.
 
---
 
## 8. `as_retriever` — turning search into a chain-ready Runnable
 
```python
retriever = vectorstore.as_retriever(
    search_type="similarity",         # or "mmr"
    search_kwargs={"k": 3},
)
retriever.invoke("What is CBT?")      # same result as similarity_search
```
 
This is what plugs directly into your LCEL chains (`{"context": retriever, ...}`).
 
---
 
## 9. Cheat sheet
 
| Operation | What it returns | Use when |
|---|---|---|
| `similarity_search` | List of `Document` | Just want the top matches |
| `similarity_search_with_score` | List of `(Document, score)` | Want to know match quality / filter weak matches |
| `similarity_search_by_vector` | List of `Document` | Already have a vector, skip re-embedding |
| `max_marginal_relevance_search` | List of `Document`, diversified | Want to avoid near-duplicate results |
| `add_documents` | — | Add new chunks without rebuilding |
| `delete` | — | Remove outdated/incorrect chunks |
| `filter={...}` param | Narrows results | Search within a specific source/category |
| `as_retriever()` | A `Retriever` (Runnable) | Plug the store into an LCEL chain |
 
---
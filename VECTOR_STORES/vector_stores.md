# Vector Stores 

## 1. What problem are we solving?
 
After splitting your documents into chunks, you need a way to **find the right chunks later** when a user asks a question. A normal database search (`WHERE text LIKE '%keyword%'`) only matches exact words — it won't know that "I feel hopeless" and "I feel like giving up" mean similar things.
 
A **vector store** solves this by searching based on *meaning*, not exact wording.
 
**Analogy:** a regular search is like Ctrl+F — exact text match only. A vector store is like asking a librarian "find me books about similar topics" — they understand the *idea*, not just the words.
 
---
 
## 2. What is a vector, really?
 
An **embedding** is a list of numbers (e.g. 1536 numbers) that represents the *meaning* of a piece of text.
 
```python
"I feel anxious"        → [0.12, -0.45, 0.88, ..., 0.03]   # 1536 numbers
"I'm feeling nervous"   → [0.14, -0.42, 0.85, ..., 0.05]   # very similar numbers!
"The stock market rose" → [-0.67, 0.21, -0.09, ..., 0.91]  # very different numbers
```
 
Texts with similar *meaning* end up with vectors that are close together in this number-space, even if they don't share any exact words. That's the whole trick.
 
**Analogy:** imagine every sentence gets a GPS coordinate based on its meaning. Similar ideas land near each other on the map; unrelated ideas land far apart. Search = "find the nearest coordinates to my question."
 
---
 
## 3. Where vector stores fit in the pipeline
 
```
Load docs → Split into chunks → Embed each chunk → Store in vector store → Retrieve → Generate
                                                        ▲ (this note)
```
 
The vector store's job: **store the embeddings + let you search them fast.**
 
---
 
## 4. How similarity search actually works
 
When a user asks a question:
 
1. The question itself gets embedded into a vector (same embedding model as your documents)
2. The vector store compares that vector against every stored chunk's vector
3. It returns the chunks whose vectors are "closest" (most similar in meaning)
**Distance/similarity metrics** (you don't need to calculate these by hand — the library does it):
- **Cosine similarity** — most common; measures the *angle* between two vectors (ignores magnitude, focuses on direction/meaning)
- **Euclidean distance** — straight-line distance between two points
- **Dot product** — another similarity measure, faster to compute
You'll mostly just pick `similarity_search` and not worry about which metric — Chroma/FAISS default to sensible choices.
 
---
 
## 5. Setting one up — ChromaDB (what you're already using)
 
```python
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
 
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
 
# Create a vector store from your chunked documents
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",   # saves to disk, survives restarts
)
 
# Search it
results = vectorstore.similarity_search("What does CBT say about catastrophizing?", k=3)
for doc in results:
    print(doc.page_content)
```
 
`k=3` means "give me the 3 most similar chunks."
 
**Loading an existing store later** (without re-embedding everything):
```python
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)
```
 
---
 
## 6. FAISS (the other one you've used)
 
```python
from langchain_community.vectorstores import FAISS
 
vectorstore = FAISS.from_documents(chunks, embeddings)
 
# Save/load to disk
vectorstore.save_local("faiss_index")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
 
results = vectorstore.similarity_search("query", k=3)
```
 
**Chroma vs FAISS — the practical difference:**
 
| | Chroma | FAISS |
|---|---|---|
| Setup | Very easy, batteries-included | Slightly more manual |
| Persistence | Built-in (`persist_directory`) | Manual save/load calls |
| Metadata filtering | Strong support | Limited |
| Best for | Most beginner/small-to-medium projects | Very large-scale, performance-critical search |
 
For a project like Solace or your Finance Advisor, Chroma is the easier and generally recommended default.
 
---
 
## 7. MMR — avoiding redundant results
 
Plain similarity search can return chunks that are all *too similar to each other* (e.g. 3 nearly-identical paragraphs). **MMR (Maximal Marginal Relevance)** balances relevance with diversity — you've already used this in your WatsonX RAG bot.
 
```python
results = vectorstore.max_marginal_relevance_search(
    "What does CBT say about catastrophizing?",
    k=3,          # final number of results
    fetch_k=10,   # how many candidates to consider before diversifying
)
```
 
**Analogy:** plain similarity search is like asking 3 friends the same question and they all give you nearly the same answer. MMR is like intentionally asking a *diverse* group so you get different angles on the topic, not repetition.
 
---
 
## 8. Turning a vector store into a Retriever
 
For use inside an LCEL chain, you convert the vector store into a `Retriever` — this is what plugs into the `{"context": retriever, ...}` pattern from your RAG notes.
 
```python
retriever = vectorstore.as_retriever(
    search_type="mmr",              # or "similarity"
    search_kwargs={"k": 3, "fetch_k": 10},
)
 
# Now it's just a Runnable, like everything else
retriever.invoke("What does CBT say about catastrophizing?")
```
 
---
 
## 9. Metadata filtering
 
Since each chunk carries `metadata` (from the loader/splitter step), you can narrow search results before/during the similarity search:
 
```python
results = vectorstore.similarity_search(
    "anxiety coping strategies",
    k=3,
    filter={"source": "cbt_handbook.pdf"},   # only search within this file
)
```
 
Useful for Solace if you eventually have multiple resource categories (e.g. CBT vs. mindfulness vs. crisis-resources) and want to search only within a specific category.
 
---
 
## 10. Adding new documents later
 
You don't need to rebuild the whole store from scratch — you can add to it incrementally:
 
```python
vectorstore.add_documents(new_chunks)
```
 
---
 
## 11. Cheat sheet
 
| Concept | What it means |
|---|---|
| Embedding | Text converted into a list of numbers representing meaning |
| Vector store | A database optimized for storing + searching embeddings |
| Similarity search | Find chunks whose vectors are closest to the query's vector |
| `k` | How many results to return |
| MMR | Similarity search that also avoids redundant/near-duplicate results |
| Retriever | A vector store wrapped as a Runnable, ready to plug into an LCEL chain |
| Metadata filter | Narrow search to specific files/categories before comparing vectors |
| Persistence | Saving the vector store to disk so you don't re-embed every run |
 
---

# Retrievers — 

## 1. What is a Retriever?

A **Retriever** is a standardized wrapper around "go fetch relevant documents for this query." It's a Runnable — same `.invoke()` interface as everything else in LCEL — so it snaps directly into a chain.

```python
retriever.invoke("What does CBT say about catastrophizing?")
# returns a list of Document objects
```

**Key distinction from a vector store:** a vector store is the *storage + raw search* layer (`similarity_search`, `add_documents`, etc). A Retriever is a *thin interface* on top of a retrieval strategy — it could be backed by a vector store, but it doesn't have to be (could be a web search, a database query, a keyword search — anything that returns documents).

**Analogy:** the vector store is the librarian's filing system. The Retriever is "hand a request to the librarian and get books back" — you don't need to know *how* they found them.

---

## 2. The simplest Retriever — wrapping a vector store

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

retriever.invoke("What does CBT say about catastrophizing?")
```

This is what you've already used — it's just `similarity_search` wrapped as a Runnable so it plugs into `{"context": retriever, ...}` chains.

---

## 3. `search_type` options

### a) `similarity` (default)
Plain nearest-neighbor search — returns the `k` closest matches.

```python
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
```

### b) `mmr` (Maximal Marginal Relevance)
Balances relevance with diversity — avoids returning several near-identical chunks.

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5},
)
```
`lambda_mult` (0–1) controls the relevance/diversity trade-off: closer to 1 = more relevance-focused, closer to 0 = more diversity-focused.

### c) `similarity_score_threshold`
Only returns results above a minimum similarity — protects against returning *irrelevant* chunks just because they were the "least bad" of the bunch.

```python
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.7, "k": 3},
)
```

**Use case for Solace:** if a user's message doesn't actually relate to anything in your CBT literature, you'd rather return *nothing* than force-feed the model a weakly related chunk.

---

## 4. MultiQueryRetriever — asking the question multiple ways

A single phrasing of a question might miss relevant chunks phrased differently. This retriever uses an LLM to generate several *variations* of the user's question, retrieves for each, and merges the results.

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=model,
)

retriever.invoke("How do I stop catastrophizing?")
# Behind the scenes, generates variations like:
# "What techniques help with catastrophic thinking?"
# "How can I manage worst-case-scenario thoughts?"
# ...then retrieves for each and combines unique results
```

**Analogy:** instead of asking one librarian one specific question, you ask 3 slightly rephrased versions and combine every unique book they hand back — covers more ground than a single exact phrasing.

**Trade-off:** more thorough, but more expensive (extra LLM call to generate the variations) and slower.

---

## 5. ContextualCompressionRetriever — trimming irrelevant parts out of retrieved chunks

Sometimes a retrieved chunk is only *partially* relevant — the rest is noise that wastes prompt space. This retriever passes retrieved chunks through a compressor (often another LLM call) that extracts just the relevant portion.

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(model)

retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(),
)
```

**Analogy:** instead of handing someone an entire page because one paragraph was relevant, you highlight and hand over just that paragraph.

---

## 6. EnsembleRetriever — combining multiple retrieval strategies

Combines results from multiple retrievers (e.g. keyword search + vector search) and merges/re-ranks them. Useful because vector search misses exact keyword/name matches sometimes (embeddings capture *meaning*, not exact terms).

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(chunks)   # classic keyword search
vector_retriever = vectorstore.as_retriever()

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5],
)
```

**Analogy:** one search is good at "find things that mean the same thing," the other is good at "find things that use the exact same words" — combining both catches more cases than either alone.

---

## 7. Self-Query Retriever — letting the LLM write the filter for you

If your documents have rich metadata (dates, categories, authors), this retriever uses an LLM to translate a natural-language question into a structured query + metadata filter automatically.

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever

# e.g. user asks: "Show me CBT techniques added after 2023"
# → LLM converts this into: query="CBT techniques", filter={"year": {"$gt": 2023}}
```

More advanced setup (needs metadata field descriptions defined), but powerful when your data has structured attributes worth filtering on.

---

## 8. Parent Document Retriever — small chunks for search, big chunks for context

Search works best on small, focused chunks. But small chunks sometimes lack enough surrounding context for the model to give a good answer. This retriever searches using small chunks, but returns the *larger parent chunk* they came from.

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=InMemoryStore(),
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
```

**Analogy:** you search using a book's index (small, precise entries), but once you find the right entry, you're handed the whole relevant chapter, not just the index line.

---

## 9. Cheat sheet

| Retriever | Solves | Trade-off |
|---|---|---|
| Basic `as_retriever()` | Standard semantic search | None — this is your default |
| `mmr` | Redundant/near-duplicate results | Slightly less "pure relevance" |
| `similarity_score_threshold` | Weak/irrelevant matches sneaking in | May return fewer than `k` results |
| `MultiQueryRetriever` | Single phrasing missing relevant chunks | Extra LLM call, slower |
| `ContextualCompressionRetriever` | Chunks with a lot of irrelevant filler | Extra LLM call per chunk |
| `EnsembleRetriever` | Vector search missing exact keyword matches | More setup, combines 2+ retrievers |
| `SelfQueryRetriever` | Need structured filtering from natural language | More setup, needs metadata schema |
| `ParentDocumentRetriever` | Small chunks lack surrounding context | More storage/setup complexity |

---

## 10. Where this fits in your RAG pipeline

```
Load docs → Split → Embed → Store → Retriever → Generate
                                        ▲ (this note)
```

The Retriever is the *interface* your chain talks to — it's what sits inside `{"context": retriever, "question": RunnablePassthrough()}`. Everything above is a different flavor of "how the retriever decides what counts as relevant."

---

## 11. One mental model to keep

**A Retriever is just "text question in, relevant Documents out" — a Runnable wrapper around a retrieval strategy.** Start with the basic `vectorstore.as_retriever()`. Only reach for the fancier ones (MMR, multi-query, ensemble, etc.) when you notice a specific problem in your actual results — like Solace returning duplicate chunks, missing keyword matches, or weak/irrelevant context.
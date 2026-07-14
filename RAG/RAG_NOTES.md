# RAG (Retrieval-Augmented Generation) -- Beginner Notes

## What is RAG?

**RAG = Retrieval + Augmented + Generation**

It is a technique where an LLM first **retrieves relevant information**
from external knowledge, then **generates an answer** using that
information.

Instead of relying only on what it learned during training, the model
can use your documents, PDFs, websites, or databases.

------------------------------------------------------------------------

## Why do we need RAG?

LLMs have limitations:

-   They may hallucinate (make up facts).
-   Their knowledge may be outdated.
-   They cannot know your private documents.

RAG solves this by giving the LLM relevant context before it answers.

------------------------------------------------------------------------

## RAG Workflow

``` text
User Query
     │
     ▼
Embedding Model
     │
     ▼
Vector Store
     │
     ▼
Retriever
     │
     ▼
Relevant Documents
     │
     ▼
LLM
     │
     ▼
Final Answer
```

------------------------------------------------------------------------

## Step 1: Documents

These are your knowledge sources.

Examples: - PDFs - Word files - Text files - Web pages - Databases

------------------------------------------------------------------------

## Step 2: Text Splitting

Large documents are divided into smaller pieces called **chunks**.

Example:

Original:

    Python is a programming language. It is used for AI, web development, and automation.

Chunks:

    Chunk 1 → Python is a programming language.
    Chunk 2 → It is used for AI, web development, and automation.

Why? - LLMs have token limits. - Smaller chunks improve retrieval
accuracy.

------------------------------------------------------------------------

## Step 3: Embeddings

Embeddings convert text into vectors (lists of numbers).

Example:

    "I love football"
    ↓

    [0.23, -0.55, 0.81, ...]

Similar meanings have similar vectors.

------------------------------------------------------------------------

## Step 4: Vector Store

A vector store stores: - Embeddings - Original text - Metadata

Popular vector databases: - FAISS - Chroma - Pinecone - Weaviate

------------------------------------------------------------------------

## Step 5: Retriever

The retriever searches the vector store and returns the most relevant
chunks.

Example:

Query:

    Who won the 2022 FIFA World Cup?

Retriever returns:

    Lionel Messi won the FIFA World Cup with Argentina in 2022.

The retriever **does not answer** the question.

------------------------------------------------------------------------

## Step 6: LLM

The retrieved context and user query are sent to the LLM.

Prompt:

    Context:
    ...

    Question:
    ...

    Answer:

The LLM generates the final response using the retrieved information.

------------------------------------------------------------------------

# Retrieval Types

## 1. Similarity Search

Returns the most similar documents.

Good for: - Basic RAG - Fast retrieval

------------------------------------------------------------------------

## 2. MMR (Max Marginal Relevance)

Returns documents that are: - Relevant - Diverse

Avoids returning many nearly identical chunks.

------------------------------------------------------------------------

## 3. MultiQuery Retriever

The LLM rewrites the user's question into multiple related queries.

Example:

Original:

    Tell me about football.

Generated queries: - soccer - FIFA - football rules - association
football

Results from all queries are combined.

Useful when users ask vague questions.

------------------------------------------------------------------------

## Common LangChain Components

-   Document Loader → Loads data
-   Text Splitter → Creates chunks
-   Embedding Model → Converts text to vectors
-   Vector Store → Stores vectors
-   Retriever → Finds relevant chunks
-   Prompt Template → Builds prompts
-   LLM → Generates answers

------------------------------------------------------------------------

## Simple RAG Pipeline

``` text
Documents
   │
   ▼
Loader
   │
   ▼
Text Splitter
   │
   ▼
Embeddings
   │
   ▼
Vector Store
   │
   ▼
Retriever
   │
   ▼
Context
   │
   ▼
LLM
   │
   ▼
Answer
```

------------------------------------------------------------------------

## Advantages

-   Reduces hallucinations
-   Uses private knowledge
-   Easy to update
-   No need to retrain the model

------------------------------------------------------------------------

## Limitations

-   Poor retrieval leads to poor answers.
-   Good chunking and embeddings are important.
-   Retrieval quality affects the final response.

------------------------------------------------------------------------

## One-line Summary

> **RAG = Retrieve relevant information first, then let the LLM generate
> an answer using that information.**

------------------------------------------------------------------------

A modern RAG pipeline is often explained in **5 stages**:

```text
Raw Documents
      │
      ▼
1. INGESTION
      │
      ▼
2. INDEXING
      │
      ▼
Vector Database
      │
User Query
      ▼
3. RETRIEVAL
      │
Relevant Chunks
      ▼
4. AUGMENTATION
      │
Prompt + Context
      ▼
5. GENERATION
      │
      ▼
Final Answer
```

---

# 1. Ingestion (Collecting Data)

**Goal:** Gather data from different sources before processing it.

Think of ingestion as **bringing books into a library**.

Common data sources:
- PDF files
- Word documents
- Text files
- Websites
- Databases
- APIs
- Notion, Google Drive, SharePoint, etc.

Example:

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("rag.pdf")
docs = loader.load()
```

**Output:** Raw `Document` objects.

---

# 2. Indexing (Preparing Knowledge)

**Goal:** Convert raw documents into a searchable knowledge base.

Steps:

```text
Documents
   │
   ▼
Text Splitter
   │
   ▼
Chunks
   │
   ▼
Embeddings
   │
   ▼
Vector Database
```

Example:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)

vectorstore = Chroma.from_documents(
    chunks,
    HuggingFaceEmbeddings()
)
```

---

# 3. Retrieval

Retrieve the most relevant chunks for a user query.

```python
retriever = vectorstore.as_retriever()

docs = retriever.invoke("What is RAG?")
```

---

# 4. Augmentation

Combine:
- User Question
- Retrieved Chunks

into a single prompt.

```python
template = """
Context:
{context}

Question:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)
```

---

# 5. Generation

The LLM reads the augmented prompt and generates the final answer.

```python
chain = prompt | llm | StrOutputParser()

response = chain.invoke({
    "context": docs,
    "question": "What is RAG?"
})
```

---

# Complete Flow

```text
Raw Documents
      │
      ▼
INGESTION
(Load from PDFs, Web, DBs, APIs)
      │
      ▼
INDEXING
(Split → Embed → Store)
      │
      ▼
Vector Database
      │
      ▼
RETRIEVAL
(Search Similar Chunks)
      │
      ▼
AUGMENTATION
(Context + Question)
      │
      ▼
GENERATION
(LLM produces answer)
```

---

# LangChain Mapping

| Stage | Purpose | Common Components |
|--------|---------|-------------------|
| Ingestion | Collect documents | Document Loaders |
| Indexing | Split, embed, store | Text Splitters, Embeddings, Chroma, FAISS |
| Retrieval | Search relevant chunks | Retriever |
| Augmentation | Build prompt | ChatPromptTemplate |
| Generation | Produce answer | Chat Model + Output Parser |

---

# Easy Analogy

- **Ingestion:** Bring books into the library.
- **Indexing:** Label and organize the books.
- **Retrieval:** Find the right books.
- **Augmentation:** Open the relevant pages beside the question.
- **Generation:** Write the answer using those pages.

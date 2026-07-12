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

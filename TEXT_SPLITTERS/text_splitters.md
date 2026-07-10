# Text Splitters 
 
**Note : these notes are generated with the help so use these for learning purpose only

## Why split text at all?
 
Once a Document Loader gives you a `Document` (say, a whole 50-page PDF as one giant blob of text), you usually can't just hand that whole thing to the model or embed it as one chunk. Two big reasons:
 
1. **Embeddings work better on small, focused chunks.** A giant chunk mixes too many topics together, so its vector becomes a blurry "average" of everything — bad for search accuracy.
2. **Models have context limits.** You can't stuff an entire book into a prompt.
So a **Text Splitter** breaks large text into smaller, manageable pieces ("chunks") before embedding and storing them.
 
**Analogy:** instead of handing someone an entire encyclopedia and asking "find me the fact about photosynthesis," you first tear it into individual paragraphs — much easier to search through, and each piece is actually about one thing.
 
---
 
## The two settings you'll always configure
 
```python
chunk_size = 1000       # max characters (or tokens) per chunk
chunk_overlap = 200     # characters repeated between consecutive chunks
```
 
**`chunk_size`** — how big each piece is. Too big → blurry embeddings, wastes context. Too small → loses context, sentence gets cut mid-thought.
 
**`chunk_overlap`** — repeats a bit of text at the boundary between chunks so you don't lose meaning if something important got split right at the cut point.
 
**Analogy for overlap:** imagine cutting a sentence exactly in half — you might lose the connection between the two halves. Overlap is like leaving a few overlapping words at each cut so nothing important falls into the gap.
 
---
 
## 1. CharacterTextSplitter — the simplest one
 
Splits purely by character count, using one separator (default: `"\n\n"`).
 
```python
from langchain_text_splitters import CharacterTextSplitter
 
splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = splitter.split_documents(docs)
```
 
**Downside:** it doesn't really understand structure — if there's no separator nearby, it just cuts wherever the character count runs out, potentially mid-sentence.
 
---
 
## 2. RecursiveCharacterTextSplitter — the one you'll actually use most
 
This is the default, recommended splitter for most use cases. It tries a *list* of separators in order of preference — paragraphs first, then sentences, then words — so it cuts at the most natural boundary it can find.
 
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
 
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""],  # tries these in order
)
chunks = splitter.split_documents(docs)
```
 
**Analogy:** it's like tearing paper along the dotted lines if they exist, and only cutting straight through as a last resort. It tries paragraph breaks first; if a paragraph is still too big, it tries line breaks; if still too big, it tries spaces (words); only as an absolute last resort does it cut mid-word.
 
**Why it's the default choice:** it keeps related sentences together as much as possible, which means each chunk stays coherent and topically focused — better for retrieval quality.
 
---
 
## 3. TokenTextSplitter — splitting by tokens, not characters
 
Models actually think in **tokens**, not characters (roughly ¾ of a word each). This splitter counts tokens directly, matching what the model will actually "see."
 
```python
from langchain_text_splitters import TokenTextSplitter
 
splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
```
 
Useful when you need precise control over context-window usage (e.g. making sure retrieved chunks + prompt + question fit within the model's limit).
 
---
 
## 4. Splitting code (bonus)
 
If you're ever chunking source code instead of prose, there's a splitter that understands code structure (functions, classes) instead of blindly cutting text.
 
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
 
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=500,
    chunk_overlap=50,
)
```
 
---
 
## 5. Cheat sheet
 
| Splitter | Splits by | Use when |
|---|---|---|
| `CharacterTextSplitter` | One fixed separator | Simple, predictable text structure |
| `RecursiveCharacterTextSplitter` | Tries multiple separators in order | Default choice — most general-purpose text |
| `TokenTextSplitter` | Token count | You need exact control over model context usage |
| `RecursiveCharacterTextSplitter.from_language()` | Code structure | Splitting source code files |
 
---
 
## 6. Where this fits in your RAG pipeline
 
```
Load docs → Split into chunks → Embed → Store in vector DB → Retrieve → Generate
              ▲ (this note)
```
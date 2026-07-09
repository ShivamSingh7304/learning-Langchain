# LangChain Runnables 

## What even is a "Runnable"?

Think of a Runnable as **a box that takes an input and gives an output**. That's it.

A prompt template? It's a box: give it variables → get a formatted prompt.
A model? It's a box: give it a prompt → get a response.
A parser? It's a box: give it raw text → get cleaned-up data.

Since every box works the same way (`.invoke(input)` → output), you can **snap them together like LEGO pieces**. That's the whole point of LCEL (LangChain Expression Language).

```python
chain = prompt | model | parser
chain.invoke({"topic": "black holes"})
```

The `|` symbol means "pass my output to the next box." Prompt's output feeds the model, model's output feeds the parser.

---

## 1. Chaining boxes in a line — the `|` operator

```python
chain = prompt | model | parser
```

This is called a `RunnableSequence` under the hood, but you'll basically never type that name — the `|` does it for you automatically.

**Analogy:** an assembly line. Each station does one job and passes the item to the next station.

---

## 2. Doing things at the same time — RunnableParallel

Sometimes you don't want one thing after another — you want two (or more) things to happen **at once**, using the same input.

```python
from langchain_core.runnables import RunnableParallel

chain = RunnableParallel(
    joke=joke_chain,
    poem=poem_chain,
)
chain.invoke({"topic": "AI"})
# {"joke": "...", "poem": "..."}
```

**Analogy:** you hand the same question to two friends at once, and get two answers back in a dictionary, labeled by name.

**Shortcut:** you don't even need to write `RunnableParallel` explicitly. Just use a plain `{ }` dictionary — LangChain converts it automatically:

```python
chain = {
    "context": retriever,
    "question": RunnablePassthrough(),
} | prompt | model | parser
```

This is the pattern almost every RAG chain uses: one branch goes and fetches relevant documents, the other branch just carries the original question forward untouched.

---

## 3. Just passing something through — RunnablePassthrough

Sometimes a box in your parallel setup shouldn't *change* the input at all — it should just hand it forward as-is.

```python
from langchain_core.runnables import RunnablePassthrough

RunnablePassthrough()   # input goes in, exact same thing comes out
```

**Why you need this:** in the RAG example above, the retriever transforms the question into "context" (documents). But you also need the *original question itself* to reach the final prompt. `RunnablePassthrough` is how you say "just forward this one unchanged."

---

## 4. Using your own Python functions — RunnableLambda

Got a normal Python function and want to slot it into a chain? Wrap it.

```python
from langchain_core.runnables import RunnableLambda

def word_count(text: str) -> int:
    return len(text.split())

counter = RunnableLambda(word_count)

chain = prompt | model | parser | counter
```

**Analogy:** you wrote your own custom LEGO piece, and `RunnableLambda` is the adapter that lets it snap into the rest of the LEGO set.

**Use it for:** cleaning up model output, adding your own logic (like checking for certain keywords), logging, formatting — anything custom.

---

## 5. Sending input down different paths — RunnableBranch

Sometimes you want: "if the input looks like X, do this chain — otherwise, do that chain."

```python
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (lambda x: "urgent" in x["text"].lower(), urgent_chain),
    (lambda x: "billing" in x["text"].lower(), billing_chain),
    default_chain,  # used if nothing above matched
)
```

**Analogy:** a fork in the road with signposts. Each signpost has a condition; the first one that matches is the road you take. If none match, you take the default road.

---

## 6. Remembering past messages — RunnableWithMessageHistory

Normally a chain forgets everything after `.invoke()` finishes. This wrapper adds memory: it automatically saves and reloads the conversation history around your chain.

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

chain_with_history.invoke(
    {"input": "I've been feeling anxious lately"},
    config={"configurable": {"session_id": "user_123"}},
)
```

**Analogy:** a notebook that automatically gets read before each conversation turn and written to after — so the chain "remembers" without you manually managing it.




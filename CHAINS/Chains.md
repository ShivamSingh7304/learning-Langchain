# LangChain Chains 
 
## What is a "Chain"?
 
A **chain** is just multiple steps linked together so the output of one step becomes the input of the next. That's the whole idea — nothing fancier than that.
 
Why do you need this? Because on its own, an LLM just takes text in and gives text out, one time. Real apps usually need *several* steps: format a prompt → call the LLM → clean up the response → maybe call the LLM again → etc. A chain is how you wire those steps together instead of writing messy glue code by hand.
 
**Analogy:** a recipe. Step 1 feeds into step 2, step 2 feeds into step 3. You don't do all steps at once — you follow the sequence.
 
# Chains: Simple, Sequential, Parallel, Conditional

Four basic shapes your chain can take. Once you recognize these four patterns, you can build almost any workflow by mixing them.

---

## 1. Simple Chain — one step

The most basic chain: prompt → model. No branching, no multiple steps.

```python
chain = prompt | model | parser
chain.invoke({"topic": "black holes"})
```

**Shape:** `input → [box] → output`

**Analogy:** asking one question, getting one answer. Nothing else happening.

**When you'd use it:** any single, self-contained task — like "summarize this text" or "classify this message's sentiment."

---

## 2. Sequential Chain — steps in a straight line

Output of step 1 feeds into step 2, output of step 2 feeds into step 3, and so on.

```python
chain = prompt_1 | model | parser | prompt_2 | model | parser
```

**Shape:** `input → [box A] → [box B] → [box C] → output`

**Analogy:** an assembly line. Each station does one job, then hands the item to the next station. Order matters — you can't skip a station or do them out of order.

**When you'd use it:** multi-step reasoning, like "extract key facts from this document" → "then write a summary from those facts" → "then translate the summary."

```python
extract_chain = extract_prompt | model | parser
summarize_chain = summarize_prompt | model | parser

full_chain = extract_chain | summarize_chain
```

---

## 3. Parallel Chain — steps at the same time

Multiple steps run on the **same input**, independently, and their results are collected together (usually as a dictionary).

```python
from langchain_core.runnables import RunnableParallel

chain = RunnableParallel(
    sentiment=sentiment_chain,
    summary=summary_chain,
)
chain.invoke({"text": "I've had a really rough week..."})
# {"sentiment": "...", "summary": "..."}
```

**Shape:**
```
        ┌─→ [box A] ─┐
input ──┤            ├─→ {results}
        └─→ [box B] ─┘
```

**Analogy:** handing the same question to two different specialists at once, then combining both of their answers.

**When you'd use it:** anytime steps don't depend on each other's output. For Solace: run sentiment analysis and RAG-context retrieval on the same user message simultaneously instead of one after another — saves time since neither needs the other's result.

**Common shortcut:** a plain `{ }` dict is auto-converted to parallel:

```python
chain = {
    "context": retriever,
    "question": RunnablePassthrough(),
} | prompt | model | parser
```

---

## 4. Conditional Chain — pick a path based on the input

Instead of always running the same steps, you check a condition and route to a *different* chain depending on the result.

```python
from langchain_core.runnables import RunnableBranch

chain = RunnableBranch(
    (lambda x: "urgent" in x["text"].lower(), urgent_chain),
    (lambda x: "billing" in x["text"].lower(), billing_chain),
    default_chain,  # runs if nothing above matched
)
```

**Shape:**
```
                ┌─→ (condition A true?) → [box A]
input → check ──┼─→ (condition B true?) → [box B]
                └─→ (else) → [default box]
```

**Analogy:** a fork in the road with signposts. You check each signpost's condition in order; the first one that matches is the path you take. If none match, you take the default path.

**When you'd use it:** for Solace, this is exactly how you'd separate "user message needs crisis-support resources" from "user message is normal conversation" — the input decides which chain runs, not a fixed sequence.

---

## 5. Combining them

Real apps usually mix all four. Example — a Solace-style flow:

```python
# Step 1: run two things in parallel on the raw message
analysis = RunnableParallel(
    sentiment=sentiment_chain,
    context=retriever,
)

# Step 2: route based on what sentiment came back
routing = RunnableBranch(
    (lambda x: x["sentiment"] == "crisis", crisis_chain),
    supportive_chat_chain,  # default
)

# Full pipeline: parallel analysis, then sequential hand-off into conditional routing
full_chain = analysis | routing
```

This reads as: **parallel** (analyze two things at once) → **sequential** (hand results to next step) → **conditional** (pick the right response chain).

# Tools in LangChain — Beginner Notes
 
## What is a Tool?
 
A **Tool** is a function the model can choose to call to do something it can't do on its own — search the web, run a calculation, look up a database, hit an API. The LLM itself can only generate text; a Tool is how you give it "hands" to actually *do* things.
 
**Analogy:** the model is a smart person answering questions from memory. A Tool is like handing them a calculator, a phone, or a search engine — they can now check facts or perform actions instead of guessing.
 
This is the foundation of **Agents** — an agent is essentially "a model + a set of tools + a loop that lets it decide which tool to call, if any."
 
---
 
## 1. Why can't the model just "do" things itself?
 
An LLM only predicts text. It doesn't have internet access, can't run code, and can't query a database — unless you explicitly give it a way to request those actions, and then run them for it. The model doesn't execute the tool itself; it just says *"I want to call this tool with these arguments"* — your code is what actually runs it and feeds the result back.
 
```
User asks → Model decides "I need a tool" → Model outputs tool name + arguments
→ Your code runs the actual tool → Result goes back to the model → Model gives final answer
```
 
---
 
## 2. Creating a simple tool — `@tool` decorator
 
The easiest way to turn a Python function into a tool:
 
```python
from langchain_core.tools import tool
 
@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)
```
 
Three things matter here:
- **The docstring** — this is what the model reads to decide *when* to use this tool. Be clear and specific.
- **Type hints** — tell the model what kind of input the tool expects.
- **Function name** — becomes the tool's name the model refers to.
```python
get_word_length.invoke({"word": "hello"})   # 5
```
 
---
 
## 3. A tool with real-world logic
 
```python
@tool
def get_weather(city: str) -> str:
    """Returns the current weather for a given city."""
    weather_data = {
        "Paris": "18°C, cloudy",
        "Tokyo": "24°C, sunny",
        "Delhi": "32°C, hazy",
    }
    return weather_data.get(city, "Weather data not available for this city.")
```
 
The model would call this automatically if a user asks something like "what's the weather in Tokyo?" — instead of guessing an answer from training data.
 
---
 
## 4. Tools with multiple inputs — use Pydantic for clarity
 
For tools with more than one argument, defining an explicit schema helps the model understand exactly what's expected.
 
```python
from pydantic import BaseModel, Field
from langchain_core.tools import tool
 
class BookFlightInput(BaseModel):
    destination: str = Field(description="City the user wants to fly to")
    passengers: int = Field(description="Number of passengers")
 
@tool(args_schema=BookFlightInput)
def book_flight(destination: str, passengers: int) -> str:
    """Books a flight to the given destination for the given number of passengers."""
    return f"Booked a flight to {destination} for {passengers} passenger(s)."
```
 
---
 
## 5. Binding tools to a model
 
Once you have tools defined, you attach them to the model so it *knows they exist* and can choose to call them.
 
```python
from langchain_groq import ChatGroq
 
model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
model_with_tools = model.bind_tools([get_word_length, get_weather])
 
response = model_with_tools.invoke("How long is the word 'elephant'?")
print(response.tool_calls)
# [{'name': 'get_word_length', 'args': {'word': 'elephant'}, 'id': '...'}]
```
 
Notice: the model doesn't return the answer directly — it returns a **tool call request**. Your code has to actually execute it.
 
---
 
## 6. Actually running the tool and feeding the result back
 
```python
from langchain_core.messages import HumanMessage, ToolMessage
 
messages = [HumanMessage("How long is the word 'elephant'?")]
ai_response = model_with_tools.invoke(messages)
messages.append(ai_response)
 
for tool_call in ai_response.tool_calls:
    # Look up and run the matching tool
    selected_tool = {"get_word_length": get_word_length}[tool_call["name"]]
    tool_output = selected_tool.invoke(tool_call["args"])
 
    # Feed the result back as a ToolMessage
    messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))
 
# Now call the model again with the tool result included
final_response = model_with_tools.invoke(messages)
print(final_response.content)
```
 
**This loop — ask model → get tool call → run tool → feed result back → ask model again — is literally what an Agent automates for you.** Frameworks like LangGraph handle this loop automatically so you don't write it by hand every time.
 
---
 
## 7. Built-in tools you don't have to write yourself
 
LangChain ships several ready-made tools for common needs:
 
```python
from langchain_community.tools import DuckDuckGoSearchRun
 
search = DuckDuckGoSearchRun()
search.invoke("latest advancements in renewable energy")
```
 
Other common built-ins: Wikipedia lookup, Python REPL (run code), SQL database query tools, requests/API-calling tools. Check `langchain_community.tools` for what's available before writing your own.
 
---
 
## 8. Cheat sheet
 
| Concept | What it means |
|---|---|
| `@tool` | Turns a Python function into something the model can call |
| Docstring | The model's *only* way of knowing what the tool does and when to use it |
| `args_schema` | Pydantic model defining structured, multi-field input |
| `bind_tools([...])` | Attaches tools to a model so it can request to use them |
| `response.tool_calls` | The model's request to call a tool (name + arguments) |
| `ToolMessage` | How you feed a tool's result back into the conversation |
| Agent | The automated loop: model decides → tool runs → result returns → model responds |
 
---
 
## 9. One mental model to keep
 
**A tool is just a function with a description attached — the description is how the model decides whether and when to use it.** The model never runs the tool itself; it only asks. Your code is the one that actually executes it and reports back. Agents are just this ask → run → report-back loop, automated.
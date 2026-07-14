from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

print(multiply.invoke({"a": 10, "b": 5}))
from langchain_core.tools import tool

@tool
def currency_converter(amount: float, from_currency: str, to_currency: str) -> str:
    """
    Convert currency from one type to another.
    """

    rates = {
        ("USD", "INR"): 83,
        ("INR", "USD"): 0.012,
        ("USD", "EUR"): 0.92,
    }

    rate = rates.get((from_currency.upper(), to_currency.upper()))

    if rate:
        return f"{amount} {from_currency} = {amount * rate:.2f} {to_currency}"

    return "Exchange rate not found."


from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

llm_with_tools = llm.bind_tools([currency_converter])


query = "Convert 100 USD to INR"

response = llm_with_tools.invoke(query)

print(response)

tool_call = response.tool_calls[0]

result = currency_converter.invoke(tool_call)

print(result)
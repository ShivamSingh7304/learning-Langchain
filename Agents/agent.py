from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


agent = create_agent(
    model=model,
    tools=[search],
    system_prompt="You are a helpful travel assistant."
)

response = agent.invoke({"messages": [{
                "role": "user",
                "content": "fifa world cup finalist team"
            }]})

response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "fifa world cup 2026 finalist team"
        }
    ]
})

print(response["messages"][-1].content)
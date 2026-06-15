from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage , HumanMessage ,AIMessage
# Load .env variables
load_dotenv()

# Create model
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

chat_history=[
    SystemMessage(content="YOU ARE A AI ASSISTANT ")
]

while True:
    user_input= input('YOU: ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result=model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("MODEL: ",result.content)

print(chat_history)


from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load .env variables
load_dotenv()

# Create model
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

chat_history=[]

while True:
    user_input= input('YOU: ')
    chat_history.append(user_input)
    if user_input == 'exit':
        break
    result=model.invoke(chat_history)
    chat_history.append(result.content)
    print("MODEL: ",result.content)

print(chat_history)







# Ask question
#response = llm.invoke("Explain AI simply")

# Print response
#print(response.content)
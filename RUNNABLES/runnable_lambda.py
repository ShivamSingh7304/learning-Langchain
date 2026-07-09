from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough , RunnableLambda
from dotenv import load_dotenv 

load_dotenv()

def word_count(text):
    words = text.split()
    return len(words)


model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
)

prompt = PromptTemplate(
    template="generate me a joke about following topic: {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

runnable_lambda = RunnableLambda(word_count)

joke_chain = RunnableSequence(prompt, model, parser)

chain = RunnableParallel(
    joke=RunnablePassthrough(),
    count=runnable_lambda
)

final_chain = joke_chain | chain

result = final_chain.invoke({"topic": "programming"})

final_result = """{} \n word count: {}""".format(result['joke'], result['count'])

print(final_result)

print(final_chain.get_graph().print_ascii())


'''  
The similar code can be written in a more compact way as below:
chain = RunnableParallel(
    joke=RunnablePassthrough(),
    count=RunnableLambda(x: len(x.split()))
)
'''
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

# Pydantic Model
class FeedbackSentiments(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        ..., description="The sentiment of the feedback"
    )

parser = PydanticOutputParser(pydantic_object=FeedbackSentiments)

# Models
model1 = ChatGroq(
    model="llama-3.3-70b-versatile",   
    temperature=0,
)

model2 = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)

str_parser = StrOutputParser()

# Prompts
classifier_prompt = PromptTemplate(
    template="""
You are a sentiment classifier.

Classify the following feedback as one of:
- positive
- negative
- neutral

Feedback:
{feedback}

{format_instructions}

Return ONLY the JSON object.
""",
    input_variables=["feedback"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)

positive_prompt = PromptTemplate(
    template="Write an empathetic response to this positive feedback:\n\n{feedback}",
    input_variables=["feedback"],
)

negative_prompt = PromptTemplate(
    template="Write an empathetic response to this negative feedback:\n\n{feedback}",
    input_variables=["feedback"],
)

neutral_prompt = PromptTemplate(
    template="Write an empathetic response to this neutral feedback:\n\n{feedback}",
    input_variables=["feedback"],
)

# Classifier
classifier_chain = classifier_prompt | model1 | parser

# Preserve original feedback
chain = RunnableParallel(
    feedback=RunnablePassthrough(),
    sentiment=classifier_chain,
)

branch_chain = RunnableBranch(

    (
        lambda x: x["sentiment"].sentiment == "positive",

        RunnableLambda(
            lambda x: {"feedback": x["feedback"]["feedback"]}
        )
        | positive_prompt|model2|str_parser,
    ),
    (
        lambda x: x["sentiment"].sentiment == "negative",
        RunnableLambda(
            lambda x: {"feedback": x["feedback"]["feedback"]}
        )
        | negative_prompt| model2| str_parser,
    ),

    (
        lambda x: x["sentiment"].sentiment == "neutral",
        RunnableLambda(
            lambda x: {"feedback": x["feedback"]["feedback"]}
        )
        | neutral_prompt| model2| str_parser,
    ),

    RunnableLambda(lambda _: "Invalid sentiment"),
)

final_chain = chain | branch_chain

result = final_chain.invoke(
    {"feedback": "The product is amazing! I love it."}
)

print(result)

final_chain.get_graph().print_ascii()
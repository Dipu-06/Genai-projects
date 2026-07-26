from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_groq import ChatGroq

load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

cricket_prompt = PromptTemplate(
    template="Generate a short, exciting joke about cricket: {topic}",
    input_variables=["topic"],
)

coding_prompt = PromptTemplate(
    template="Generate a short, nerdy joke about programming/coding: {topic}",
    input_variables=["topic"],
)

general_prompt = PromptTemplate(
    template="Generate a general funny joke about: {topic}",
    input_variables=["topic"],
)

parser = StrOutputParser()

cricket_chain = cricket_prompt | model | parser
coding_chain = coding_prompt | model | parser
general_chain = general_prompt | model | parser

router_branch = RunnableBranch(
    (lambda x: x.get("topic", "").strip().lower() == "cricket", cricket_chain),
    (lambda x: x.get("topic", "").strip().lower() == "coding", coding_chain),
    general_chain,  # Default fallback chain if no conditions match
)

def word_count(text):
    return len(text.split())

chain = (
    router_branch
    | RunnableLambda(
        lambda joke: {"joke": joke, "word_count": word_count(joke)}
    )
)

print("--- CRICKET TOPIC ---")
print(chain.invoke({"topic": "cricket"}))

print("\n--- CODING TOPIC ---")
print(chain.invoke({"topic": "coding"}))

print("\n--- GENERAL TOPIC ---")
print(chain.invoke({"topic": "elephants"}))
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_groq import ChatGroq

load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

def word_count(text):
    return len(text.split())

prompt1 = PromptTemplate(
    template="generate a joke on {topic}", input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="explain why this joke is funny: {joke}", input_variables=["joke"]
)

parser = StrOutputParser()

joke_chain = prompt1 | model | parser

analysis_chain = RunnableParallel(
    {
        "joke": RunnablePassthrough(),
        "explanation": prompt2 | model | parser,
        "count": RunnableLambda(word_count),
    }
)

complete_chain = joke_chain | analysis_chain

result = complete_chain.invoke({"topic": "cricket"})

print(f"**Joke:** {result['joke']}\n")
print(f"**Word Count:** {result['count']}\n")
print(f"**Explanation:** {result['explanation']}")
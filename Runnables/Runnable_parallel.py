from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
model=ChatGroq(model="llama-3.1-8b-instant", temperature=0) 
prompt1=PromptTemplate(
    template="write a tweet on {topic}",
    input_variables=['topic']
) 
prompt2=PromptTemplate(
    template="write a linkedinpost on {topic}",
    input_variables=['topic']
) 
parser=StrOutputParser()
chain=RunnableParallel({
    'tweet':prompt1|model|parser,
    'post':prompt2|model|parser
})
result=chain.invoke({'topic':"genai"})
print(result)
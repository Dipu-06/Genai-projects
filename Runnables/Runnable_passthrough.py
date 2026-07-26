from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
model=ChatGroq(model="llama-3.1-8b-instant", temperature=0) 
prompt1=PromptTemplate(
    template="generate  a joke on {topic}",
    input_variables=['topic']
) 
prompt2=PromptTemplate(
    template="explain the joke  {joke}",
    input_variables=['joke']
) 
parser=StrOutputParser()
chain=RunnableParallel({
'topic':RunnablePassthrough(),
'joke':RunnableSequence(prompt2,model,parser)

})
result=chain.invoke({'topic':"cricket"})
print(result)

from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
llm=HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    timeout=120
)
model=ChatHuggingFace(llm=llm)
template1=PromptTemplate(
    template="write a detail report on {topic}",
    input_variables=['topic']
    
)
template2=PromptTemplate(
    template="write a five line summary on{text}",
    input_variables=['text']
    
)
Parser=StrOutputParser()
chain=template1|model|Parser|template2|model|Parser
result=chain.invoke({'topic':"blackhole"})
print(result)
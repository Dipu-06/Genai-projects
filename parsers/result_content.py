from  langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
load_dotenv()
llm=HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
    
)
model=ChatHuggingFace(llm=llm)
template1=PromptTemplate(
    template="write a report on {topic}",
    input_variables=['topic']
)
template2=PromptTemplate(
    template="write a five line summary on {text}",
    input_variables=['text']
)
prompt1=template1.invoke({'topic':"blackhole"})
result=model.invoke(prompt1)
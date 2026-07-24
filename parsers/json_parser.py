from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
load_dotenv()
llm=HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    timeout=120
)
model=ChatHuggingFace(llm=llm)
Parser=JsonOutputParser()
template1=PromptTemplate(
    template="give me name ,age,city  of a fictional person\n{format_instructions}",
    input_variables=[],
    partial_variables={'format_instructions':Parser.get_format_instructions()}
    
)


chain=template1|model|Parser
result=chain.invoke({})
print(result)
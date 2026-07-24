from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
load_dotenv()
llm=HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    timeout=120
)
model=ChatHuggingFace(llm=llm)

# 1. Define your schema using Pydantic
class FactList(BaseModel):
    fact1: str = Field(description="Fact 1 about the topic")
    fact2: str = Field(description="Fact 2 about the topic")
    fact3: str = Field(description="Fact 3 about the topic")
    fact4: str = Field(description="Fact 4 about the topic")
    fact5: str = Field(description="Fact 5 about the topic")

# 2. Instantiate the parser
parser = PydanticOutputParser(pydantic_object=FactList)
template=PromptTemplate(
    template="generate name,age and coity of fictional person{city}\n{format_instructions}",
    input_variables=['city'],
    partial_variables={'format_instructions': parser.get_format_instructions()}

)
chain=template|model|parser
result=chain.invoke({'city':"india"})
print(result)
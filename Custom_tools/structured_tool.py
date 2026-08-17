from langchain_core.tools import StructuredTool
from pydantic import BaseModel,Field
class Multiply(BaseModel):
    a:int=Field(required=True,description="first no")
    b:int=Field(required=True,description="second no")
def multiply(a:int,b:int):
        return a*b
multiply_tool=StructuredTool.from_function(
    func=multiply,
    name="multiply",
    description="multiply two nos",
    args_schema=Multiply
)
result=multiply_tool.invoke({"a":3,"b":5})
print(result)
print(multiply_tool.name)

     
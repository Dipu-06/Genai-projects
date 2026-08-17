from langchain_core.tools import tool
def multiply(a,b):
    '''multiplies the two numbers'''
    return a*b
def multiply(a:int,b:int):
    '''multiplies two numbers'''
    return a*b
@tool
def multiply(a:int,b:int):
    '''multiplies two numbers'''
    return a*b
result=multiply.invoke({"a":3,"b":7})
print(result)
print(multiply.args)
print(multiply.description)
print(multiply.name)
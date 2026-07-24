from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
llm=HuggingFaceEndpoint(
repo_id="Qwen/Qwen2.5-7B-Instruct",
task="text-generation",
timeout=120
)
model=ChatHuggingFace(llm=llm)
template=PromptTemplate(
    template="write five lines on{topic}",
    input_variables=['topic']
)
parser=StrOutputParser()
chain=template|model|parser
result=chain.invoke({'topic':"spring"})
print(result)
chain.get_graph().print_ascii()

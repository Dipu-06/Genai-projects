from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
# Import ResponseSchema & StructuredOutputParser from 'langchain.output_parsers'
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

load_dotenv()

# 1. ceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    timeout=120
)Initialize LLM
llm = HuggingFa
model = ChatHuggingFace(llm=llm)

# 2. Define Response Schemas
schema = [
    ResponseSchema(name='fact1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact3', description='Fact 3 about the topic'),
    ResponseSchema(name='fact4', description='Fact 4 about the topic'),
    ResponseSchema(name='fact5', description='Fact 5 about the topic')
]

# 3. Create parser using .from_response_schemas()
parser = StructuredOutputParser.from_response_schemas(schema)

# 4. Define Prompt matching the schemas
template1 = PromptTemplate(
    template="Give me 5 interesting facts about {topic}.\n{format_instructions}",
    input_variables=["topic"],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

# 5. Build and invoke chain
chain = template1 | model | parser
result = chain.invoke({"topic": "Black Holes"})

print(result)
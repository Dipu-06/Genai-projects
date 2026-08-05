from langchain_text_splitters import RecursiveCharacterTextSplitter
text="my name is dipali, i am 22 years old. i have a home"
 

splitter=RecursiveCharacterTextSplitter(
    chunk_size=5,
    chunk_overlap=0,
    
)
result=splitter.split_text(text)
print(result)
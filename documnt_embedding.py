from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
documents = [
    "The capital city of India is New Delhi.",
    "Python is a versatile programming language widely used in AI development.",
    "A standard soccer match is played over 90 minutes across two halves.",
    "The solar system consists of eight planets orbiting the sun."
]
query = "Tell me about tech stacks for artificial intelligence"
doc_embedding=embedding.embed_documents(documents)
query_embedding=embedding.embed_query(query)
scores=cosine_similarity([query_embedding],doc_embedding)
print(list(enumerate(scores)))
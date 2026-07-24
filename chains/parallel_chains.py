import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
model1 = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
model2 = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
template1=PromptTemplate(
    template="give notes on {text}",
    input_variables=['text']
)
template2=PromptTemplate(
    template="write 5 question and ans from{text}",
    input_variables=['text']
)
template3=PromptTemplate(
    template="merge the give two  documets notes\n{notes}and quiz\n{quiz}",
    input_variables=['notes','quiz']
)
parser=StrOutputParser()
parallel_chain=RunnableParallel({
    'notes':template1|model1|parser,
    'quiz':template2|model2|parser
})
merge_chain=template3|model1|parser
chain=parallel_chain|merge_chain
text="A Decision Tree is a popular, intuitive machine learning algorithm used for both classification and regression tasks. It models decisions in a tree-like structure, starting from a single root node representing the entire dataset, which then splits into internal decision nodes (representing features or tests) and eventually terminates in leaf nodes (representing final predictions or class labels). At each step, the algorithm selects the feature that best splits the data into the most distinct, homogenous subsets—often evaluated using metric criteria like Gini Impurity or Information Gain (Entropy). Because decision trees closely mirror human decision-making, they are easy to interpret, require minimal data preprocessing, and can naturally handle both numerical and categorical data. However, deep trees are prone to overfitting (memorizing training data rather than generalizing), which is typically mitigated by pruning the tree or using ensemble methods like Random Forests."
result=chain.invoke({'text':text})
print(result)
chain.get_graph().print_ascii()

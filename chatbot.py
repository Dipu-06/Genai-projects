import os

import streamlit as st

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_core.messages import (
    HumanMessage,
    AIMessage
)

# -----------------------------
# Load API Key
# -----------------------------

load_dotenv()



# -----------------------------
# Hugging Face Model
# -----------------------------

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-8B",
    task="text-generation",
    
)

chat_model = ChatHuggingFace(llm=llm)

# -----------------------------
# Prompt
# -----------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a friendly AI Assistant.

        Answer politely.

        Explain everything simply.

        If the user asks coding questions,
        provide Python examples.
        """
    ),

    MessagesPlaceholder(variable_name="history"),

    ("human", "{input}")
])

chain = prompt | chat_model

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖"
)

st.title("🤖 AI Chatbot")

# -----------------------------
# Session State
# -----------------------------

if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# Display Chat History
# -----------------------------

for message in st.session_state.history:

    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)

    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# -----------------------------
# User Input
# -----------------------------

user_input = st.chat_input("Ask me anything...")

if user_input:

    with st.chat_message("user"):
        st.markdown(user_input)

    response = chain.invoke({
        "history": st.session_state.history,
        "input": user_input
    })

    with st.chat_message("assistant"):
        st.markdown(response.content)

    st.session_state.history.append(
        HumanMessage(content=user_input)
    )

    st.session_state.history.append(
        AIMessage(content=response.content)
    )
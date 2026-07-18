import os
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_core.messages import (
    HumanMessage,
    AIMessage
)

st.set_page_config(
    page_title="My AI Chatbot",
    page_icon="💬"
)

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-8B",
    task="text-generation",
)
chat_model = ChatHuggingFace(llm=llm)
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a friendly AI Assistant.
        Answer politely.
        Explain everything simply.
        If the user asks coding questions, provide Python examples.
        """
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])
chain = prompt | chat_model
with st.sidebar:
    st.title("🤖 Chat Settings")
    st.write("Use the button below to reset the conversation.")
    if st.button("Clear Chat"):
        st.session_state.history = []
        st.rerun()

st.title("💬 My AI Chatbot")
st.write("Welcome! Ask me anything below.")

if "history" not in st.session_state:
    st.session_state.history = []

if len(st.session_state.history) == 0:
    st.info("Hello! How can I help you today?")

for message in st.session_state.history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(message.content)

user_input = st.chat_input("Type your message here...")

if user_input:
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)
    st.session_state.history.append(HumanMessage(content=user_input))

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            response = chain.invoke({
                "history": st.session_state.history,
                "input": user_input
            })
            st.markdown(response.content)

    st.session_state.history.append(AIMessage(content=response.content))
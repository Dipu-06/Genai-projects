import re
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- BACKEND MODEL LOGIC ---

def extract_video_id(url: str) -> str:
    """Robustly extracts the 11-character YouTube video ID from any URL format."""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_youtube_transcript_safely(youtube_url: str):
    """Fetches transcript directly using youtube-transcript-api to prevent 400 errors."""
    video_id = extract_video_id(youtube_url)
    if not video_id:
        raise ValueError("Could not determine a valid YouTube video ID from the provided URL.")
    
    try:
        # Fetch transcript list supporting English and Hindi
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'hi'])
        transcript_text = "\n".join([item['text'] for item in transcript_list])
        return f"YouTube Video ({video_id})", transcript_text
    except Exception as e:
        raise RuntimeError(f"YouTube blocked automated fetch. Details: {e}")


def ask_youtube_chatbot(transcript_text: str, chat_history: list, user_question: str, openai_api_key: str):
    """Answers user queries based on the transcript as a conversational chatbot."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=openai_api_key)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert AI tutor assistant. Answer the user's questions strictly based on the provided YouTube video transcript context. If the answer is not in the transcript, politely say you couldn't find it in the video."),
        ("human", "Transcript Context:\n{context}\n\nChat History:\n{history}\n\nUser Question: {question}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-5:]])
    
    response = chain.invoke({
        "context": transcript_text,
        "history": history_str,
        "question": user_question
    })
    
    return response


# --- STREAMLIT UI ---

st.set_page_config(page_title="YouTube Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 YouTube Video Chatbot")
st.write("Chat with any YouTube video securely using its transcript!")

# Sidebar configuration for API key
with st.sidebar:
    st.header("Configuration")
    api_key_input = st.text_input("OpenAI API Key", type="password")
    st.markdown("---")
    if st.button("Clear Chat Session"):
        st.session_state.messages = []
        st.session_state.transcript = None
        st.session_state.video_title = None
        st.rerun()

# Input choice: URL or Direct Transcript Fallback
input_method = st.radio("Choose Input Method:", ["YouTube URL", "Paste Transcript Text Directly"])

transcript = None
video_title = None

if input_method == "YouTube URL":
    url_input = st.text_input("Enter YouTube Video URL:")
    if url_input and api_key_input:
        if "transcript" not in st.session_state or st.session_state.get("current_source") != url_input:
            with st.spinner("Fetching transcript from video..."):
                try:
                    title, transcript_data = get_youtube_transcript_safely(url_input)
                    st.session_state.transcript = transcript_data
                    st.session_state.video_title = title
                    st.session_state.current_source = url_input
                    st.session_state.messages = [{"role": "assistant", "content": f"Loaded video successfully! What would you like to know about it?"}]
                except Exception as e:
                    st.error(f"{e}\n\n💡 *Tip: Switch to 'Paste Transcript Text Directly' mode above if YouTube blocks this link.*")
else:
    manual_transcript = st.text_area("Paste the full YouTube transcript text here:")
    if manual_transcript and api_key_input:
        if "transcript" not in st.session_state or st.session_state.get("current_source") != manual_transcript:
            st.session_state.transcript = manual_transcript
            st.session_state.video_title = "Manually Provided Transcript"
            st.session_state.current_source = manual_transcript
            st.session_state.messages = [{"role": "assistant", "content": "Loaded manual transcript successfully! What would you like to know?"}]

# Initialize chat messages history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat interface if transcript is loaded
if "transcript" in st.session_state and st.session_state.transcript:
    st.success(f"Active Source: **{st.session_state.get('video_title')}**")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about this content..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = ask_youtube_chatbot(
                        st.session_state.transcript,
                        st.session_state.messages,
                        prompt,
                        api_key_input
                    )
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"An error occurred: {e}")
elif not api_key_input:
    st.warning("Please enter your OpenAI API key in the sidebar to proceed.")
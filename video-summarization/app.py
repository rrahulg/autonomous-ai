import os
import time
import tempfile
from pathlib import Path

import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if API_KEY:
    genai.configure(api_key=API_KEY)

st.set_page_config(
    page_title="MultiModal AI Agent",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Video Summarization with MultiModal AI Agent")
st.header("Powered by Gemini AI 🚀")


@st.cache_resource
def initialise_agent():
    return Agent(
        name="AI Video Summarizer",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True,
    )


agent = initialise_agent()

video_file = st.file_uploader(
    "Upload a video file",
    type=["mp4", "mov", "avi"],
    help="Upload a video for AI analysis",
)

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name

    st.video(video_path, format="video/mp4", start_time=0)

    user_query = st.text_area(
        "What insights are you seeking from the video?",
        placeholder="Ask anything about the video content. AI will analyze and gather context.",
        help="Provide specific questions for analysis.",
    )

    if st.button("🔍 Analyze Video", key="analyze_video_button"):
        if not user_query:
            st.warning("Please enter a question or insight to analyze the video.")
        else:
            try:
                with st.spinner("Processing video and gathering insights..."):

                    analysis_prompt = f"""
                        The user has uploaded a video. While direct video analysis is not supported,
                        please generate insights based on video metadata and user-provided questions.

                        User Query: {user_query}

                        Provide an informative and actionable response.
                    """

                    response = agent.run(analysis_prompt)

                st.subheader("Analysis Result")
                st.markdown(response.content)

            except Exception as error:
                st.error(f"An error occurred during analysis: {error}")
            finally:
                Path(video_path).unlink(missing_ok=True)
else:
    st.info("Upload a video file to begin analysis.")

st.markdown(
    """
    <style>
    .stTextArea textarea {
        height: 100px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

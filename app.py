import streamlit as st
from st_audiorec import st_audiorec
from pathlib import Path
import tempfile
import os
from utils import upload_to_github

from QwenIELTSEvaluator import QwenIELTSEvaluator
from dotenv import load_dotenv

load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_OWNER = os.getenv("REPO_OWNER")
REPO_NAME = os.getenv("REPO_NAME")
TARGET_FOLDER = os.getenv("TARGET_FOLDER")

evaluator = QwenIELTSEvaluator(api_key=DASHSCOPE_API_KEY)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="IELTS Speaking Evaluation", layout="wide")

# Sidebar instructions
with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    1. Read the IELTS question displayed.
    2. Record your answer using the microphone below.
    3. Upload your recording to GitHub.
    4. The system will evaluate your response and provide feedback.
    """)

QUESTION = "Describe a time when you helped someone. What happened?"

# Main content
st.title("🎤 IELTS Speaking Evaluation")
st.markdown(f"**Question:** {QUESTION}")

st.markdown("### 🎙️ Record Your Answer")
audio_bytes = st_audiorec()

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    
    # Save temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        audio_path = tmp.name

    st.success("Recording captured!")

    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("⬆️ Upload to GitHub"):
            github_url = upload_to_github(audio_path, f"user_recording_{os.path.basename(audio_path)}")
            if github_url:
                st.success(f"✅ Uploaded! [Click to play]({github_url})")
                st.session_state["github_url"] = github_url
    with col2:
        if st.button("🔍 Evaluate Answer"):
            if "github_url" not in st.session_state:
                st.warning("⚠️ Please upload your recording first.")
            else:
                st.subheader("⏳ Evaluating your answer...")
                try:
                    model_feedback = evaluator.evaluate_audio(st.session_state["github_url"])
                    st.markdown("### 📊 Model Feedback")
                    st.markdown(model_feedback)
                except Exception as e:
                    st.error(f"Evaluation error: {e}")

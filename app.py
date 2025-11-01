# =======================================================
# 🎤 AI INTERVIEW ASSESSMENT APP - WhisperX + SentenceTransformer
# =======================================================

# ----------- Instalasi otomatis bila belum ada (hindari error FFmpeg) -----------
import os
import subprocess
import sys

# Install dependensi penting jika belum ada
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
except FileNotFoundError:
    print("Installing FFmpeg...")
    subprocess.call(["apt-get", "install", "-y", "ffmpeg"])

for pkg in ["streamlit", "torch", "whisperx", "sentence-transformers"]:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        install(pkg)

# =======================================================
# 📚 IMPORT LIBRARY
# =======================================================
import streamlit as st
import tempfile
import whisperx
import torch
from sentence_transformers import SentenceTransformer, util
import random

# =======================================================
# 🧠 CONFIGURASI UTAMA
# =======================================================
st.set_page_config(page_title="AI Interview - Speech to Text", page_icon="🎤")

st.title("🎤 AI Interview Assessment")
st.write(
    "Welcome to your **virtual interview session**. "
    "Please record your video response to the question below. "
    "Our AI will automatically transcribe and analyze your answer for relevance."
)

# =======================================================
# 📋 Langkah 1 — Pertanyaan Otomatis
# =======================================================
questions = [
    "Tell me about yourself and your professional background.",
    "Why do you want to work for our company?",
    "Describe a challenge you faced and how you overcame it.",
    "What are your strengths and weaknesses?",
    "Where do you see yourself in five years?"
]
question = random.choice(questions)
st.subheader("🎯 Interview Question")
st.info(question)

# =======================================================
# 🎥 Langkah 2 — Upload Rekaman Video Jawaban
# =======================================================
st.subheader("🎬 Upload Your Answer")
video_file = st.file_uploader(
    "🎥 Upload your interview video (MP4, MOV, AVI):", type=["mp4", "mov", "avi"]
)

if video_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name

    st.video(video_path)

    # =======================================================
    # 🔊 Langkah 3 — Ekstrak Audio
    # =======================================================
    st.write("🎧 Extracting audio from video...")
    audio_path = video_path.replace(".mp4", ".wav")

    command = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # =======================================================
    # 🧠 Langkah 4 — Transkripsi dengan WhisperX
    # =======================================================
    st.write("🧠 Running WhisperX for transcription...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisperx.load_model("small", device)
    result = model.transcribe(audio_path)

    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    aligned_result = whisperx.align(
        result["segments"], model_a, metadata, audio_path, device
    )

    transcript_text = aligned_result["text"]

    # =======================================================
    # 📜 Langkah 5 — Tampilkan Hasil Transkripsi
    # =======================================================
    st.subheader("📝 Transcription Result")
    st.success(transcript_text)

    # =======================================================
    # 📊 Langkah 6 — Analisis Relevansi Jawaban (AI)
    # =======================================================
    st.subheader("🤖 AI Answer Relevance Analysis")

    model_st = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings_question = model_st.encode(question, convert_to_tensor=True)
    embeddings_answer = model_st.encode(transcript_text, convert_to_tensor=True)

    similarity = util.pytorch_cos_sim(embeddings_question, embeddings_answer).item()
    similarity_percentage = similarity * 100

    if similarity_percentage >= 80:
        verdict = "Excellent — Your answer is highly relevant to the question."
    elif similarity_percentage >= 60:
        verdict = "Good — Your answer is somewhat relevant but could be more focused."
    else:
        verdict = "Needs Improvement — Your answer doesn't strongly relate to the question."

    st.metric("Relevance Score", f"{similarity_percentage:.2f}%")
    st.info(verdict)

    # =======================================================
    # 📊 (Opsional) Statistik Bicara
    # =======================================================
    st.subheader("📈 Speech Summary")
    word_count = len(transcript_text.split())
    st.write(f"Total words spoken: **{word_count} words**")

    st.success("✅ Analysis complete! You can review your results above.")

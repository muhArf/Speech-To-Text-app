import streamlit as st
import tempfile
import whisper
import torch
import os
import random
from sentence_transformers import SentenceTransformer, util
import imageio_ffmpeg as ffmpeg_lib
import subprocess

# ===========================
# 🧠 CONFIG
# ===========================
st.set_page_config(page_title="AI Interview - Speech to Text", page_icon="🎤")

st.title("🎤 Virtual AI Interview")
st.write(
    "Selamat datang di sesi wawancara virtual! 🎬 "
    "Silakan jawab pertanyaan yang diberikan dengan merekam video jawabanmu. "
    "Sistem AI kami akan menyalin ucapanmu menjadi teks dan "
    "menganalisis kesesuaian jawaban terhadap pertanyaan."
)

# ===========================
# 📋 Langkah 1 — Pertanyaan Otomatis
# ===========================
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

# ===========================
# 🎥 Langkah 2 — Upload Video
# ===========================
st.subheader("🎬 Upload Your Answer")
video_file = st.file_uploader("📹 Upload video kamu (MP4/MOV/AVI):", type=["mp4", "mov", "avi"])

if video_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name

    # ===========================
    # 🔊 Ekstraksi Audio (tanpa ffmpeg system)
    # ===========================
    st.write("🎧 Mengekstrak audio dari video...")

    audio_path = video_path.replace(".mp4", ".wav")
    ffmpeg_binary = ffmpeg_lib.get_ffmpeg_exe()

    try:
        command = [
            ffmpeg_binary, "-i", video_path, "-vn",
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path, "-y"
        ]
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        st.success("✅ Audio berhasil diekstrak.")
    except Exception as e:
        st.error(f"❌ Gagal mengekstrak audio: {e}")
        st.stop()

    # ===========================
    # 🧠 Transkripsi Whisper (bukan WhisperX)
    # ===========================
    st.write("🧠 Menjalankan model Whisper...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("small", device=device)
    result = model.transcribe(audio_path)

    st.subheader("📝 Hasil Transkripsi")
    st.success(result["text"])

    # ===========================
    # 📊 Analisis
    # ===========================
    st.subheader("📊 Analisis Sederhana")
    word_count = len(result["text"].split())
    st.write(f"Jumlah kata: **{word_count}**")

    st.subheader("💡 Analisis Kesesuaian Jawaban")
    model_st = SentenceTransformer("all-MiniLM-L6-v2")

    question_emb = model_st.encode(question, convert_to_tensor=True)
    answer_emb = model_st.encode(result["text"], convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(question_emb, answer_emb).item() * 100

    st.write(f"Tingkat relevansi: **{similarity:.2f}%**")

    if similarity > 75:
        st.success("✅ Jawaban sangat relevan.")
    elif similarity > 50:
        st.warning("⚠️ Cukup relevan, bisa lebih fokus.")
    else:
        st.error("❌ Jawaban kurang relevan.")

    st.success("🎉 Analisis selesai!")

import streamlit as st
import whisperx
import torch
import tempfile
import os
import subprocess
from moviepy.editor import VideoFileClip
from sentence_transformers import SentenceTransformer, util

# ===== Title & Description =====
st.title("🎤 AI Interview Assessment System")
st.write("""
Selamat datang di sistem wawancara otomatis berbasis AI.  
Unggah video jawaban wawancara kamu, dan sistem akan menganalisis **kecocokan jawaban dengan pertanyaan** menggunakan model AI WhisperX & Sentence Transformer.
""")

# ===== Upload video =====
uploaded_video = st.file_uploader("🎥 Unggah video jawaban kamu (format: mp4, mov, avi)", type=["mp4", "mov", "avi"])

# ===== Input pertanyaan =====
question = st.text_input("❓ Masukkan pertanyaan wawancara", "Tell me about yourself")

# ===== Proses saat video diunggah =====
if uploaded_video is not None:
    st.video(uploaded_video)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_video.read())
        video_path = temp_video.name

    # ===== Ekstrak audio dari video =====
    st.info("🎧 Mengekstrak audio dari video...")
    try:
        audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, codec="pcm_s16le")
        st.success("✅ Audio berhasil diekstrak!")
    except Exception as e:
        st.error(f"❌ Gagal mengekstrak audio: {e}")
        st.stop()

    # ===== Load WhisperX =====
    st.info("🧠 Memproses transkripsi dengan WhisperX (ini mungkin memakan waktu)...")
    try:
        model = whisperx.load_model("small", device="cpu")
        result = model.transcribe(audio_path)
        transcript = result["text"]
        st.success("✅ Transkripsi selesai!")
        st.text_area("📜 Hasil Transkripsi:", transcript, height=200)
    except Exception as e:
        st.error(f"❌ Gagal melakukan transkripsi: {e}")
        st.stop()

    # ===== Analisis Kecocokan Jawaban =====
    st.info("🤖 Menganalisis relevansi jawaban dengan pertanyaan...")
    try:
        model_st = SentenceTransformer('all-MiniLM-L6-v2')
        emb_q = model_st.encode(question, convert_to_tensor=True)
        emb_a = model_st.encode(transcript, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(emb_q, emb_a).item()
        score = round(similarity * 100, 2)

        if score >= 80:
            feedback = "Jawaban sangat relevan dengan pertanyaan. 👍"
        elif score >= 60:
            feedback = "Jawaban cukup relevan, tetapi bisa lebih fokus. 🧐"
        else:
            feedback = "Jawaban kurang relevan dengan pertanyaan. 🚫"

        st.subheader("📊 Hasil Analisis:")
        st.metric("Tingkat Relevansi", f"{score}%")
        st.write(feedback)
    except Exception as e:
        st.error(f"❌ Gagal menganalisis jawaban: {e}")

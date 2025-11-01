import streamlit as st
import whisperx
import torch
import ffmpeg
import tempfile
import os
from sentence_transformers import SentenceTransformer, util

# -----------------------------
# App Title
# -----------------------------
st.set_page_config(page_title="AI Interview Speech Analyzer", page_icon="🎤", layout="centered")
st.title("🎤 AI Interview Speech Analyzer")
st.write("Unggah video wawancara kandidat untuk analisis otomatis berdasarkan pertanyaan dan isi jawabannya.")

# -----------------------------
# Input Pertanyaan dan Video
# -----------------------------
question = st.text_area("🧩 Masukkan pertanyaan wawancara (dalam Bahasa Inggris):")
uploaded_video = st.file_uploader("🎬 Unggah video kandidat (format MP4, MOV, AVI, dll)", type=["mp4", "mov", "avi"])

# -----------------------------
# Jalankan Analisis
# -----------------------------
if st.button("🚀 Jalankan Analisis"):

    if not question:
        st.warning("⚠️ Harap isi pertanyaan terlebih dahulu.")
        st.stop()

    if uploaded_video is None:
        st.warning("⚠️ Harap unggah video terlebih dahulu.")
        st.stop()

    # Simpan video sementara
    st.info("📁 Menyimpan video sementara...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
        tmp_video.write(uploaded_video.read())
        video_path = tmp_video.name

    # Ekstraksi audio pakai ffmpeg
    st.info("🎧 Mengekstrak audio dari video...")
    try:
        audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        (
            ffmpeg
            .input(video_path)
            .output(audio_path, acodec='pcm_s16le', ac=1, ar='16k')
            .run(quiet=True, overwrite_output=True)
        )
        st.success("✅ Audio berhasil diekstrak!")
    except Exception as e:
        st.error(f"❌ Gagal mengekstrak audio: {e}")
        st.stop()

    # Load model WhisperX
    st.info("🤖 Memuat model WhisperX...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisperx.load_model("small", device)
    except Exception as e:
        st.error(f"❌ Gagal memuat model WhisperX: {e}")
        st.stop()

    # Transkripsi audio
    st.info("📝 Melakukan transkripsi jawaban kandidat...")
    try:
        result = model.transcribe(audio_path)
        candidate_text = result["text"].strip()
        st.success("✅ Transkripsi selesai!")
        st.text_area("🗣️ Hasil Transkripsi:", candidate_text, height=200)
    except Exception as e:
        st.error(f"❌ Gagal melakukan transkripsi: {e}")
        st.stop()

    # Analisis kesesuaian jawaban dengan pertanyaan
    st.info("🧠 Menganalisis kesesuaian jawaban dengan pertanyaan...")
    try:
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = embedder.encode([question, candidate_text], convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
        score = round(similarity * 100, 2)

        st.success(f"💡 Hasil Analisis: {score}% relevansi antara pertanyaan dan jawaban.")
        if score > 75:
            st.write("🟢 Jawaban sangat relevan dengan pertanyaan.")
        elif score > 50:
            st.write("🟡 Jawaban cukup relevan namun masih perlu perbaikan.")
        else:
            st.write("🔴 Jawaban kurang relevan dengan pertanyaan.")
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan dalam analisis kesesuaian: {e}")

    # Hapus file sementara
    try:
        os.remove(video_path)
        os.remove(audio_path)
    except:
        pass

    st.success("🎯 Analisis selesai!")


# =======================================================
# 🧰 AUTO SETUP - PASTIKAN FFMPEG TERINSTAL
# =======================================================
import subprocess

try:
    subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
except FileNotFoundError:
    import os
    os.system("apt-get update && apt-get install -y ffmpeg")

# =======================================================
# 🎤 AI INTERVIEW APP
# =======================================================
import streamlit as st
import tempfile
import whisperx
import torch
import os
import random
from sentence_transformers import SentenceTransformer, util

# ===========================
# 🧠 CONFIGURASI UTAMA
# ===========================
st.set_page_config(page_title="AI Interview - Speech to Text", page_icon="🎤")

st.title("🎤 Virtual AI Interview")
st.write(
    "Selamat datang di sesi wawancara virtual! 🎬 "
    "Silakan jawab pertanyaan yang diberikan dengan merekam video jawabanmu. "
    "Sistem AI kami akan secara otomatis menyalin ucapanmu menjadi teks dan "
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
# 🎥 Langkah 2 — Rekam atau Upload Video
# ===========================
st.subheader("🎬 Record or Upload Your Answer")
video_file = st.file_uploader("📹 Upload video jawaban kamu (format MP4, MOV, AVI, dll):", type=["mp4", "mov", "avi"])

if video_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name

    # ===========================
    # 🔊 Langkah 3 — Ekstrak Audio dari Video
    # ===========================
    st.write("🎧 Mengekstrak audio dari video...")
    audio_path = video_path.replace(".mp4", ".wav")

    command = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # ===========================
    # 🧠 Langkah 4 — Transkripsi dengan WhisperX
    # ===========================
    st.write("🧠 Menjalankan model WhisperX untuk transkripsi...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisperx.load_model("small", device)

    result = model.transcribe(audio_path)

    # Alignment (penyelarasan waktu)
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    aligned_result = whisperx.align(
        result["segments"], model_a, metadata, audio_path, device
    )

    # ===========================
    # 📜 Langkah 5 — Hasil Transkripsi
    # ===========================
    st.subheader("📝 Hasil Transkripsi")
    st.success(aligned_result["text"])

    # ===========================
    # 📊 Analisis Sederhana
    # ===========================
    st.subheader("📊 Analisis Sederhana")
    word_count = len(aligned_result["text"].split())
    st.write(f"Jumlah kata yang diucapkan: **{word_count} kata**")

    duration_command = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", audio_path
    ]
    duration = float(subprocess.check_output(duration_command).decode("utf-8").strip())
    wpm = (word_count / duration) * 60
    st.write(f"Kecepatan bicara: **{wpm:.1f} kata per menit**")

    # ===========================
    # 💬 Analisis Kesesuaian Jawaban dengan Pertanyaan
    # ===========================
    st.subheader("💡 Analisis Kesesuaian Jawaban")

    model_st = SentenceTransformer("all-MiniLM-L6-v2")

    # Embedding pertanyaan & jawaban
    question_emb = model_st.encode(question, convert_to_tensor=True)
    answer_emb = model_st.encode(aligned_result["text"], convert_to_tensor=True)

    similarity = util.pytorch_cos_sim(question_emb, answer_emb).item() * 100

    st.write(f"Tingkat relevansi jawaban terhadap pertanyaan: **{similarity:.2f}%**")

    if similarity > 75:
        st.success("✅ Jawaban kamu sangat relevan dengan pertanyaan!")
    elif similarity > 50:
        st.warning("⚠️ Jawaban kamu cukup relevan, tetapi bisa lebih spesifik.")
    else:
        st.error("❌ Jawaban kamu kurang relevan dengan pertanyaan. Coba jawab lebih fokus.")

    st.success("🎉 Proses selesai! Lihat hasil analisis jawabanmu di atas.")

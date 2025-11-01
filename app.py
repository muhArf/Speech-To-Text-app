import streamlit as st
import tempfile
import subprocess
import whisperx
import torch
import os

# ===========================
# 🧠 CONFIGURASI UTAMA
# ===========================
st.set_page_config(page_title="AI Interview - Speech to Text", page_icon="🎤")

st.title("🎤 AI Interview Assessment - Speech to Text (WhisperX)")
st.write("Rekam jawaban wawancara kamu secara langsung, dan sistem akan otomatis menyalin ucapanmu menjadi teks menggunakan **WhisperX AI**.")

# ===========================
# 📋 Langkah 1 — Pertanyaan Otomatis
# ===========================
import random

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
# 🎥 Langkah 2 — Rekam Video
# ===========================
st.subheader("🎬 Record Your Answer")
video_file = st.camera_input("Tekan tombol di bawah untuk mulai merekam jawaban kamu:")

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
        "-vn",  # tanpa video
        "-acodec", "pcm_s16le",  # format wav
        "-ar", "16000", "-ac", "1",
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

    # Alignment (penyelarasan waktu agar lebih akurat)
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    aligned_result = whisperx.align(
        result["segments"], model_a, metadata, audio_path, device
    )

    # ===========================
    # 📜 Langkah 5 — Tampilkan Hasil Transkripsi
    # ===========================
    st.subheader("📝 Hasil Transkripsi")
    st.success(aligned_result["text"])

    # ===========================
    # 📊 (Opsional) Analisis Sederhana
    # ===========================
    st.subheader("📊 Analisis Sederhana")
    word_count = len(aligned_result["text"].split())
    st.write(f"Jumlah kata yang diucapkan: **{word_count} kata**")

    duration_command = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1", audio_path]
    duration = float(subprocess.check_output(duration_command).decode("utf-8").strip())
    wpm = (word_count / duration) * 60
    st.write(f"Kecepatan bicara: **{wpm:.1f} kata per menit**")

    st.success("✅ Proses selesai! Kamu bisa melihat hasil dan analisis jawabanmu di atas.")

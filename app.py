import streamlit as st
import tempfile
import subprocess
import whisperx
import torch
import os
import random
from sentence_transformers import SentenceTransformer, util

# ===========================
# 🧠 CONFIGURASI UTAMA
# ===========================
st.set_page_config(page_title="AI Interview - Speech to Text", page_icon="🎤")

st.title("🎤 AI Interview Simulation")
st.write(
    "Welcome to your virtual interview session. "
    "Please answer the following question by recording or uploading your video response. "
    "Our AI system will automatically analyze your speech, transcribe it into text, "
    "and evaluate how relevant your answer is to the interview question."
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
# 🎥 Langkah 2 — Upload Video Jawaban
# ===========================
st.subheader("🎬 Record or Upload Your Answer")
video_file = st.file_uploader(
    "🎥 Upload your interview video (MP4, MOV, AVI):",
    type=["mp4", "mov", "avi"]
)

if video_file is not None:
    # Simpan file video sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name

    st.video(video_file)

    # ===========================
    # 🔊 Ekstraksi Audio dari Video
    # ===========================
    st.write("🎧 Extracting audio from video...")
    audio_path = video_path.replace(".mp4", ".wav")

    try:
        command = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            audio_path
        ]
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        st.success("✅ Audio extracted successfully!")

        # ===========================
        # 🧠 Transkripsi dengan WhisperX
        # ===========================
        st.write("🧠 Running WhisperX for transcription...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisperx.load_model("small", device)

        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio)

        # Alignment untuk hasil lebih akurat
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], device=device
        )
        aligned_result = whisperx.align(
            result["segments"], model_a, metadata, audio_path, device
        )

        # ===========================
        # 📜 Hasil Transkripsi
        # ===========================
        st.subheader("📝 Transcription Result")
        transcription_text = aligned_result["text"].strip()
        st.text_area("Transcribed Text:", transcription_text, height=250)

        # ===========================
        # 📊 Analisis Dasar
        # ===========================
        st.subheader("📊 Basic Analysis")
        word_count = len(transcription_text.split())

        # Hitung durasi audio (detik)
        try:
            duration_command = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", audio_path
            ]
            duration_output = subprocess.check_output(duration_command).decode("utf-8").strip()
            duration = float(duration_output) if duration_output else 1.0
        except Exception:
            duration = 1.0

        wpm = (word_count / duration) * 60
        st.write(f"🗣️ Word count: **{word_count}**")
        st.write(f"⏱️ Speaking speed: **{wpm:.1f} words per minute**")

        # ===========================
        # 🧩 Analisis Relevansi Jawaban
        # ===========================
        st.subheader("🧠 Relevance Analysis")

        try:
            model_semantic = SentenceTransformer('all-MiniLM-L6-v2')

            emb_question = model_semantic.encode(question, convert_to_tensor=True)
            emb_answer = model_semantic.encode(transcription_text, convert_to_tensor=True)

            similarity = util.pytorch_cos_sim(emb_question, emb_answer).item()
            st.write(f"Relevance Score: **{similarity:.2f}**")

            # Interpretasi skor
            if similarity > 0.75:
                st.success("✅ The answer is highly relevant to the interview question.")
            elif similarity > 0.5:
                st.warning("⚠️ The answer is somewhat relevant but could be more focused.")
            else:
                st.error("❌ The answer is not relevant to the question.")
        except Exception as e:
            st.error(f"Relevance analysis failed: {e}")

        st.success("✅ Interview processing complete! You can review the results above.")

    except FileNotFoundError:
        st.error("⚠️ FFmpeg not found. Please install FFmpeg on the server.")
    except subprocess.CalledProcessError as e:
        st.error(f"❌ FFmpeg processing error: {e}")
    except Exception as e:
        st.error(f"Unexpected error during processing: {e}")

    finally:
        # Bersihkan file sementara
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)

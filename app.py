import streamlit as st
import speech_recognition as sr
import difflib
import random
import tempfile
from pydub import AudioSegment
import os

# ---------------------------------------------------------
# 🎨 Konfigurasi Tampilan
# ---------------------------------------------------------
st.set_page_config(page_title="AI Job Interview Assessment", page_icon="🎤", layout="centered")

st.markdown("""
    <style>
    .main {
        background-color: #f9fafb;
        font-family: 'Poppins', sans-serif;
        color: #222;
    }
    .title {
        text-align: center;
        font-size: 2em;
        font-weight: 600;
        color: #2563eb;
        margin-bottom: 0.2em;
    }
    .subtitle {
        text-align: center;
        font-size: 1.1em;
        color: #555;
        margin-bottom: 1.5em;
    }
    .card {
        background-color: white;
        padding: 1.3em 1.6em;
        border-radius: 1em;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.1);
        margin-top: 1em;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🤖 AI Job Interview Assessment</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Latih kemampuan speaking Anda dengan simulasi wawancara kerja berbasis AI</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 🧠 Soal Interview (Random)
# ---------------------------------------------------------
questions = [
    "Can you tell me about yourself?",
    "Why do you want to work at our company?",
    "What are your greatest strengths?",
    "What are your weaknesses and how do you improve them?",
    "Describe a time when you faced a challenge at work and how you handled it.",
    "Where do you see yourself in five years?",
    "Can you describe your ideal work environment?",
    "Tell me about a successful project you worked on.",
    "How do you handle stress or pressure?",
    "Why should we hire you?",
    "Describe a situation where you worked in a team successfully.",
    "What motivates you to do your best work?",
    "Tell me about a mistake you made and what you learned from it.",
    "What do you know about our company?",
    "What are your salary expectations?"
]

question = random.choice(questions)
st.markdown(f"<div class='card'><b>💼 Interview Question:</b><br>{question}</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 🎧 Upload Audio
# ---------------------------------------------------------
st.markdown("### 🎙️ Upload your recorded answer (any audio format)")

audio_file = st.file_uploader("Upload your answer", type=["wav", "mp3", "m4a", "ogg", "flac", "aac"])

if audio_file is not None:
    st.audio(audio_file)

    recognizer = sr.Recognizer()
    text_result = ""

    try:
        # Simpan file sementara dengan format aslinya
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as temp_input:
            temp_input.write(audio_file.read())
            temp_input.flush()

            # Konversi ke WAV sementara
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
                sound = AudioSegment.from_file(temp_input.name)
                sound.export(temp_wav.name, format="wav")

                with sr.AudioFile(temp_wav.name) as source:
                    audio_data = recognizer.record(source)
                    with st.spinner("🎧 Converting your speech to text..."):
                        text_result = recognizer.recognize_google(audio_data)
                        st.success("✅ Transcription Complete!")
                        st.markdown(f"<div class='card'><b>🗣️ Your Answer:</b><br>{text_result}</div>", unsafe_allow_html=True)

    except sr.UnknownValueError:
        st.error("❌ Sorry, we couldn't understand your audio. Please try again.")
    except sr.RequestError:
        st.error("⚠️ Speech Recognition service unavailable.")
    except Exception as e:
        st.error(f"⚠️ Error processing audio: {e}")

    # ---------------------------------------------------------
    # 💯 Penilaian Jawaban
    # ---------------------------------------------------------
    if text_result:
        st.markdown("### 💯 AI Evaluation Result")

        reference_answer = "I am a motivated and hardworking person who enjoys learning new skills and contributing to team success."

        similarity = difflib.SequenceMatcher(None, text_result.lower(), reference_answer.lower()).ratio()
        score = round(similarity * 100, 2)

        st.markdown(f"""
        <div class='card'>
            <b>🎯 Similarity Score:</b> {score}/100 <br><br>
            <b>💬 Sample Strong Answer:</b><br>
            {reference_answer}
        </div>
        """, unsafe_allow_html=True)

        if score >= 85:
            st.success("🌟 Excellent! You spoke clearly and answered with strong confidence.")
        elif score >= 70:
            st.info("👍 Good job! Try to use more details and clear pronunciation.")
        elif score >= 50:
            st.warning("🗣️ Fair attempt. Focus on structure and clarity in your response.")
        else:
            st.error("😕 Try again with clearer pronunciation and complete sentences.")

# ---------------------------------------------------------
# 📘 Instructions
# ---------------------------------------------------------
st.markdown("""
---
**📋 Instructions:**
1️⃣ Read the interview question carefully.  
2️⃣ Record your answer using any format (.mp3, .m4a, .wav, .ogg, etc).  
3️⃣ Upload your audio file above.  
4️⃣ The AI will transcribe and score your speaking performance.  

*Powered by Streamlit, SpeechRecognition, pydub, and Google Speech API.*
""")

import streamlit as st
import speech_recognition as sr
import difflib
import random
import tempfile

# ---------------------------------------------------------
# 🎓 Konfigurasi Tampilan
# ---------------------------------------------------------
st.set_page_config(page_title="AI Speech-to-Text Assessment", page_icon="🎤", layout="centered")

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

st.markdown("<div class='title'>🎤 AI Speech-to-Text Assessment</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Uji kemampuan speaking Anda menggunakan AI Speech Recognition</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 🧠 Soal Speaking (Random)
# ---------------------------------------------------------
questions = [
    "Describe your favorite holiday destination and why you like it.",
    "Tell me about your favorite hobby and why you enjoy it.",
    "What is your opinion about online learning?",
    "Describe a memorable event in your life.",
    "If you could change one thing about the world, what would it be?",
    "Who is someone that inspires you and why?",
    "What are your goals for the next five years?",
    "Describe your daily routine on weekends.",
    "What kind of movies do you like to watch and why?",
    "How do you usually spend your free time?"
]

# Pilih satu soal acak
question = random.choice(questions)

st.markdown(f"<div class='card'><b>📝 Soal:</b><br>{question}</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 🎧 Upload Audio
# ---------------------------------------------------------
st.markdown("### 📤 Upload jawaban audio Anda (format **.wav** saja)")

audio_file = st.file_uploader("Pilih file audio", type=["wav"])

if audio_file is not None:
    st.audio(audio_file)

    recognizer = sr.Recognizer()

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            temp_wav.write(audio_file.read())
            temp_wav.flush()

            with sr.AudioFile(temp_wav.name) as source:
                audio_data = recognizer.record(source)
                with st.spinner("🎧 Sedang mengonversi audio ke teks..."):
                    text_result = recognizer.recognize_google(audio_data)
                    st.success("✅ Transkripsi Berhasil!")
                    st.markdown(f"<div class='card'><b>🗣️ Hasil Transkripsi:</b><br>{text_result}</div>", unsafe_allow_html=True)

    except sr.UnknownValueError:
        st.error("❌ Audio tidak dapat dikenali. Silakan coba lagi.")
        text_result = ""
    except sr.RequestError:
        st.error("⚠️ Gagal terhubung ke layanan Speech Recognition.")
        text_result = ""
    except Exception as e:
        st.error("⚠️ Pastikan file audio dalam format .wav (PCM).")

    # ---------------------------------------------------------
    # 💯 Penilaian Jawaban
    # ---------------------------------------------------------
    if 'text_result' in locals() and text_result:
        st.markdown("### 💯 Hasil Penilaian")

        reference_answer = "My favorite hobby is playing football because it keeps me active and helps me relax."

        similarity = difflib.SequenceMatcher(None, text_result.lower(), reference_answer.lower()).ratio()
        score = round(similarity * 100, 2)

        st.markdown(f"""
        <div class='card'>
            <b>🎯 Skor Kecocokan:</b> {score} / 100 <br><br>
            <b>💬 Jawaban Referensi:</b><br>
            {reference_answer}
        </div>
        """, unsafe_allow_html=True)

        if score >= 85:
            st.success("🌟 Excellent speaking! Pronunciation and content are very clear.")
        elif score >= 70:
            st.info("👍 Good! You can improve by speaking more clearly.")
        elif score >= 50:
            st.warning("🗣️ Fair. Try to answer with more complete sentences.")
        else:
            st.error("😕 Low accuracy. Please record again with clearer pronunciation.")

# ---------------------------------------------------------
# 📘 Petunjuk
# ---------------------------------------------------------
st.markdown("""
---
**Petunjuk:**
1️⃣ Baca soal di atas dengan suara jelas.  
2️⃣ Rekam jawaban Anda dan upload file audio **berformat .WAV (PCM 16-bit)**.  
3️⃣ Sistem akan otomatis menilai hasil transkripsi Anda.  

*Dibangun dengan Python, Streamlit, dan SpeechRecognition.*
""")

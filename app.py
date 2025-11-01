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
# 🧠 Soal Interview dan Jawaban Ideal
# ---------------------------------------------------------
questions_and_answers = {
    "Can you tell me about yourself?":
        "I am a dedicated and motivated professional with a passion for learning and growing in my career. I enjoy working in team environments and continuously improving my skills.",
    "Why do you want to work at our company?":
        "I admire your company’s commitment to innovation and employee development. I believe my skills align with your values and I want to contribute to your ongoing success.",
    "What are your greatest strengths?":
        "My greatest strengths are adaptability, teamwork, and problem-solving. I always stay calm under pressure and look for effective solutions.",
    "What are your weaknesses and how do you improve them?":
        "One of my weaknesses is that I tend to be a perfectionist, but I’ve learned to balance quality with efficiency by setting realistic deadlines.",
    "Describe a time when you faced a challenge at work and how you handled it.":
        "Once, I faced a tight project deadline. I organized the team, delegated tasks clearly, and we successfully completed the project ahead of time.",
    "Where do you see yourself in five years?":
        "In five years, I see myself growing into a leadership role where I can guide and support others while continuing to develop my professional skills.",
    "Can you describe your ideal work environment?":
        "My ideal work environment is one that encourages collaboration, open communication, and continuous learning.",
    "Tell me about a successful project you worked on.":
        "In my last job, I led a small team to improve a workflow system that reduced processing time by 20%. It was rewarding to see our teamwork pay off.",
    "How do you handle stress or pressure?":
        "I stay organized, prioritize tasks, and take short breaks when needed. I view pressure as an opportunity to perform better.",
    "Why should we hire you?":
        "You should hire me because I bring a mix of relevant skills, passion, and a proactive attitude that will add value to your team.",
    "Describe a situation where you worked in a team successfully.":
        "In a previous project, I worked with a diverse team where I learned to communicate effectively and ensure everyone’s strengths were utilized for success.",
    "What motivates you to do your best work?":
        "I am motivated by challenges and the opportunity to make a meaningful impact through my work.",
    "Tell me about a mistake you made and what you learned from it.":
        "I once miscommunicated a deadline with a colleague, but I learned to confirm expectations early to prevent similar issues in the future.",
    "What do you know about our company?":
        "Your company is known for its innovative approach and strong focus on customer satisfaction, which aligns with my professional goals.",
    "What are your salary expectations?":
        "I am open to discussing a salary that reflects the role’s responsibilities and aligns with industry standards."
}

question = random.choice(list(questions_and_answers.keys()))
sample_answer = questions_and_answers[question]

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
        # Simpan file sementara (format asli)
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as temp_input:
            temp_input.write(audio_file.read())
            temp_input.flush()

            # Konversi ke WAV
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
                sound = AudioSegment.from_file(temp_input.name)
                sound.export(temp_wav.name, format="wav")

                # Transkripsi audio
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

        similarity = difflib.SequenceMatcher(None, text_result.lower(), sample_answer.lower()).ratio()
        score = round(similarity * 100, 2)

        st.markdown(f"""
        <div class='card'>
            <b>🎯 Similarity Score:</b> {score}/100 <br><br>
            <b>💬 Sample Strong Answer:</b><br>
            {sample_answer}
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
2️⃣ Record your answer in any format (.mp3, .m4a, .wav, etc).  
3️⃣ Upload your audio file above.  
4️⃣ The AI will transcribe and evaluate your speaking performance.  

*Powered by Streamlit, SpeechRecognition, pydub, and Google Speech API.*
""")

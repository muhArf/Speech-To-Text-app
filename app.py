import streamlit as st
import whisper
import subprocess
import json
import tempfile
import os
import numpy as np

# ============================================================
#   CONFIGURASI APLIKASI STREAMLIT
# ============================================================
st.title("Aplikasi Transkripsi Interview – Whisper + Streamlit")
st.write("Upload **5 video sesuai urutan pertanyaan** kemudian proses transkripsi.")

# Membuat folder kerja
os.makedirs("videos", exist_ok=True)
os.makedirs("audios", exist_ok=True)
os.makedirs("transcripts", exist_ok=True)
os.makedirs("dataset", exist_ok=True)

# ============================================================
#   UPLOAD VIDEO (5 FILE)
# ============================================================
uploaded_files = st.file_uploader(
    "Upload 5 video interview (q1–q5)", 
    type=["mp4", "mov", "mkv"], 
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f"{len(uploaded_files)} file berhasil diupload.")
else:
    st.stop()

# Pastikan 5 file
if len(uploaded_files) != 5:
    st.error("HARUS upload tepat **5 file video**!")
    st.stop()

# ============================================================
#   SIMPAN VIDEO SECARA LOKAL
# ============================================================
video_paths = []
for i, file in enumerate(uploaded_files, start=1):
    video_path = f"videos/q{i}.mp4"
    with open(video_path, "wb") as f:
        f.write(file.read())
    video_paths.append(video_path)

st.info("Semua video tersimpan.")

# ============================================================
#   FUNGSI KONVERSI VIDEO → AUDIO
# ============================================================
def video_to_audio(video_path, audio_path):
    cmd = [
        "ffmpeg", "-i", video_path, "-vn",
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_path, "-y"
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

# ============================================================
#   KONVERSI SEMUA VIDEO KE AUDIO
# ============================================================
st.subheader("Konversi Video → Audio")
audio_paths = []

for i in range(1, 6):
    video_path = f"videos/q{i}.mp4"
    audio_path = f"audios/q{i}.wav"
    video_to_audio(video_path, audio_path)
    audio_paths.append(audio_path)
    st.write(f"Converted q{i}.mp4 → q{i}.wav")

# ============================================================
#   LOAD MODEL WHISPER
# ============================================================
st.subheader("Load Model Whisper")
model = whisper.load_model("medium")
st.success("Model berhasil dimuat!")

# ============================================================
#   TRANSKRIP AUDIO → TEKS
# ============================================================
st.subheader("Transkripsi Audio")
transcripts = {}

for i in range(1, 6):
    audio = f"audios/q{i}.wav"
    st.write(f"Mentranskrip q{i}.wav ...")

    result = model.transcribe(audio, fp16=False)
    text = result["text"]

    transcripts[i] = text

    with open(f"transcripts/q{i}.txt", "w") as f:
        f.write(text)

st.success("Semua audio selesai ditranskrip.")

# ============================================================
#   DATA PERTANYAAN
# ============================================================
QUESTIONS = {
    1: "Can you share any specific challenges you faced while working on certification and how you overcame them?",
    2: "Can you describe your experience with transfer learning in TensorFlow? How did it benefit your projects?",
    3: "Describe a complex TensorFlow model you have built and the steps you took to ensure its accuracy and efficiency.",
    4: "Explain how to implement dropout in a TensorFlow model and the effect it has on training.",
    5: "Describe the process of building a convolutional neural network (CNN) using TensorFlow for image classification."
}

# ============================================================
#   BENTUKKAN RAW DATASET
# ============================================================
dataset_raw = []

for i in range(1, 6):
    dataset_raw.append({
        "question_id": i,
        "question_text": QUESTIONS[i],
        "transcript": transcripts[i],
        "score": None,
        "reason": None
    })

with open("dataset/dataset_raw.json", "w") as f:
    json.dump(dataset_raw, f, indent=2)

st.success("dataset_raw.json berhasil dibuat.")

# ============================================================
#   FORM PENILAIAN MANUAL
# ============================================================
st.subheader("Penilaian Jawaban Kandidat")

labeled = []

for item in dataset_raw:
    st.write("### Pertanyaan", item["question_id"])
    st.write(item["question_text"])
    st.write("**Transcript:**")
    st.info(item["transcript"])

    score = st.slider(f"Skor untuk Q{item['question_id']} (0–4)", 0, 4)
    reason = st.text_input(f"Alasan skoring Q{item['question_id']}")

    item["score"] = score
    item["reason"] = reason

    labeled.append(item)

# ============================================================
#   GENERATE INTERVIEW SESSION JSON
# ============================================================
if st.button("Simpan Hasil"):
    interview_score = sum(i["score"] for i in labeled)

    interview_session = {
        "candidate_id": 131,
        "answers": labeled,
        "interview_score": interview_score,
        "decision": "PASSED" if interview_score >= 12 else "Need Human",
        "overall_notes": "Generated automatically from dataset"
    }

    with open("dataset/interview_session.json", "w") as f:
        json.dump(interview_session, f, indent=2)

    st.success("interview_session.json berhasil dibuat!")

    st.download_button(
        "Download dataset_raw.json",
        data=open("dataset/dataset_raw.json", "rb").read(),
        file_name="dataset_raw.json"
    )

    st.download_button(
        "Download dataset_labeled_answers.json",
        data=json.dumps(labeled, indent=2),
        file_name="dataset_labeled_answers.json"
    )

    st.download_button(
        "Download interview_session.json",
        data=json.dumps(interview_session, indent=2),
        file_name="interview_session.json"
    )

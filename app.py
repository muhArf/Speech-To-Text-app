# app.py
import streamlit as st
from pathlib import Path
import tempfile, os, time, io, base64
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from difflib import SequenceMatcher

# Optional imports (handled gracefully)
try:
    import openai
except Exception:
    openai = None

try:
    import whisper
except Exception:
    whisper = None

try:
    from jiwer import wer
except Exception:
    wer = None

# Optional audio recorder component
# pip package: streamlit-audiorecorder
try:
    from audiorecorder import audiorecorder
    HAS_RECORD = True
except Exception:
    HAS_RECORD = False

st.set_page_config(page_title="AI Assessment — Speech-to-Text", page_icon="🧠", layout="wide")

# --- CSS Styling (professional dark glass) ---
st.markdown(
    """
    <style>
    :root{
      --bg:#0b1220; --card:#0f1724; --muted:#9fb4d9; --accent:#6ee7b7;
    }
    .stApp { background: linear-gradient(180deg,var(--bg) 0%, #041026 100%); color: #e6eef8; }
    .topbar { display:flex; gap:12px; align-items:center; }
    .brand { font-size:22px; font-weight:700; color: #ffffff; }
    .subtitle { color:var(--muted); margin-top:-6px; font-size:13px; }
    .card { background: rgba(255,255,255,0.03); padding:16px; border-radius:12px; box-shadow: 0 6px 22px rgba(2,6,23,0.6); }
    .question { font-size:16px; font-weight:600; color:#ffffff; }
    .small { font-size:12px; color:var(--muted); }
    .metric { background: rgba(255,255,255,0.02); padding:10px; border-radius:8px; }
    .center { display:flex; justify-content:center; align-items:center; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Header ---
col1, col2 = st.columns([4,1])
with col1:
    st.markdown('<div class="topbar"><div class="brand">AI Assessment • Speaking Test</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Uji kemampuan berbicara — rekam, transkrip otomatis, dan nilai berdasarkan kunci.</div>', unsafe_allow_html=True)
with col2:
    st.image("https://static.streamlit.io/images/brand/streamlit-mark-light.png", width=56)

st.write("")

# --- Sidebar: settings and instructions ---
with st.sidebar:
    st.markdown("## Pengaturan & Instruksi")
    st.markdown("1. Pilih soal, tekan **Record** atau unggah file audio jawaban kamu.\n2. Tekan **Transcribe & Score**.\n3. Lihat transkrip, skor kemiripan, dan WER (jika tersedia).")
    st.markdown("---")
    st.selectbox("Metode Transkripsi (pilih)", ["Automatic (OpenAI Whisper API)", "Local Whisper (server)", "Fallback (SpeechRecognition Google)"], index=0, key="method")
    st.selectbox("Bahasa (opsional)", ["auto", "id", "en"], index=0, key="language")
    st.checkbox("Tampilkan waveform", value=True, key="show_wave")
    st.checkbox("Tampilkan WER jika tersedia (jiwer)", value=True, key="show_wer")
    st.markdown("---")
    st.markdown("**Catatan deploy**:")
    st.markdown("- Untuk OpenAI Whisper API: tambahkan `OPENAI_API_KEY` di Secrets pada Streamlit Cloud.")
    st.markdown("- Local Whisper membutuhkan model & resources; tidak direkomendasikan di Streamlit Cloud.")
    st.markdown("- Jika tidak ada recorder, kamu dapat mengunggah file audio (.wav/.mp3).")

# --- Hardcoded questions (5) ---
QUESTIONS = [
    {"id":1, "question":"Perkenalkan dirimu secara singkat dalam 30 detik." , "reference":"Nama saya ... saya berasal dari ... dan saya bekerja sebagai ..."},
    {"id":2, "question":"Jelaskan satu proyek yang paling membanggakan yang pernah kamu kerjakan." , "reference":"Saya mengerjakan proyek ... yang berfokus pada ... hasilnya ..."},
    {"id":3, "question":"Apa tantangan terbesar yang pernah kamu hadapi dalam tim dan bagaimana kamu mengatasinya?" , "reference":"Tantangan terbesar adalah ... saya mengatasi dengan ..."},
    {"id":4, "question":"Kenapa kamu tertarik dengan posisi ini?" , "reference":"Saya tertarik karena posisi ini ... dan saya ingin ..."},
    {"id":5, "question":"Ceritakan tentang satu ide inovatif yang ingin kamu implementasikan di perusahaan." , "reference":"Ide inovatif saya adalah ... yang bertujuan untuk ..."}
]

# --- UI layout: left = exam, right = results ---
left, right = st.columns([2,3])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Soal Speaking Test")
    q_idx = st.number_input("Pilih nomor soal", min_value=1, max_value=len(QUESTIONS), value=1, step=1)
    current = QUESTIONS[q_idx-1]
    st.markdown(f'<div class="question">Soal {current["id"]}: {current["question"]}</div>', unsafe_allow_html=True)
    st.markdown('<div class="small">Rekam jawabanmu (disarankan 20–90 detik) atau unggah file audio.</div>', unsafe_allow_html=True)
    st.write("")

    # Recorder (if available)
    audio_bytes = None
    if HAS_RECORD:
        st.markdown("#### Rekam di browser")
        rec = audiorecorder("Klik untuk memulai rekaman", "Klik untuk berhenti")
        if len(rec) > 0:
            # rec is bytes-like WAV in many implementations
            audio_bytes = rec
            st.success("Rekaman berhasil direkam.")
            st.audio(audio_bytes)
    else:
        st.info("Recorder browser tidak tersedia di environment ini. Silakan unggah file audio (.wav/.mp3).")

    st.markdown("---")
    st.markdown("#### Unggah file audio (jika tidak merekam)")
    uploaded = st.file_uploader("Unggah jawaban audio", type=["wav","mp3","m4a","flac","ogg"], accept_multiple_files=False)

    # If user recorded and also uploaded, prefer recorded
    chosen_audio = None
    if audio_bytes:
        chosen_audio = audio_bytes
        audio_label = "rekaman_browser.wav"
    elif uploaded is not None:
        chosen_audio = uploaded.getvalue()
        audio_label = uploaded.name

    st.write("")
    st.markdown("---")
    if st.button("Transcribe & Score", type="primary"):
        if chosen_audio is None:
            st.warning("Belum ada audio. Rekam atau unggah file audio terlebih dahulu.")
        else:
            # Save to temp file
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            try:
                # Attempt to read bytes and convert to wav if needed using soundfile
                try:
                    # If it's bytes (from recorder), write directly
                    if isinstance(chosen_audio, bytes):
                        tf.write(chosen_audio)
                        tf.flush()
                    else:
                        # uploaded may be BytesIO
                        tf.write(chosen_audio)
                        tf.flush()
                except Exception:
                    tf.write(chosen_audio)
                    tf.flush()
                tf.close()
                audio_path = tf.name

                # Ensure consistent WAV sampling
                try:
                    y, sr = librosa.load(audio_path, sr=16000, mono=True)
                    sf.write(audio_path, y, 16000, format="WAV")
                except Exception:
                    # fallback: leave file as-is
                    pass

                st.session_state["last_audio_path"] = audio_path
                st.session_state["last_question"] = current
                st.session_state["transcribe_time"] = time.time()
                st.success("Audio siap untuk ditranskrip. Lihat panel sebelah kanan untuk hasil.")
            except Exception as e:
                st.error(f"Gagal menyiapkan audio: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Hasil Transkripsi & Penilaian")
    if "last_audio_path" not in st.session_state:
        st.info("Belum ada transkrip. Lakukan rekaman/unggah + tekan **Transcribe & Score**.")
    else:
        audio_path = st.session_state["last_audio_path"]
        question = st.session_state.get("last_question", QUESTIONS[0])

        # Show player
        try:
            with open(audio_path, "rb") as f:
                audio_bytes_player = f.read()
            st.audio(audio_bytes_player)
        except Exception:
            st.warning("Gagal memutar audio (format tidak dikenali).")

        # Waveform
        if st.session_state.get("show_wave", True):
            try:
                y, sr = librosa.load(audio_path, sr=16000, mono=True)
                duration = len(y) / sr
                fig, ax = plt.subplots(figsize=(8,2))
                times = np.linspace(0, duration, num=len(y))
                ax.plot(times, y, linewidth=0.4)
                ax.set_xlim(0, duration)
                ax.set_yticks([])
                ax.set_xlabel("Seconds")
                ax.set_title("Waveform")
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Gagal membuat waveform: {e}")

        # Transcription (choose method)
        method = st.sidebar.get("method")
        language = st.sidebar.get("language", "auto")
        show_wer_flag = st.sidebar.get("show_wer", True)

        def transcribe_with_openai(audio_filepath, language="auto"):
            if openai is None:
                raise RuntimeError("openai package belum terpasang di environment.")
            key = os.getenv("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY") if "OPENAI_API_KEY" in st.secrets else None)
            if not key:
                raise RuntimeError("OPENAI_API_KEY tidak ditemukan. Tambahkan ke Secrets atau environment.")
            openai.api_key = key
            with open(audio_filepath, "rb") as af:
                try:
                    res = openai.Audio.transcribe("whisper-1", af, language=None if language=="auto" else language)
                    text = res.get("text") if isinstance(res, dict) else str(res)
                except Exception as e:
                    raise RuntimeError(f"Transcribe gagal (OpenAI): {e}")
            return text

        def transcribe_local_whisper(audio_filepath, language="auto"):
            if whisper is None:
                raise RuntimeError("whisper package tidak terpasang.")
            model = whisper.load_model("small")
            opts = {}
            if language != "auto":
                opts["language"] = language
            result = model.transcribe(audio_filepath, **opts)
            return result.get("text","")

        def transcribe_fallback_google(audio_filepath, language="en-US"):
            try:
                import speech_recognition as sr
            except Exception:
                raise RuntimeError("speech_recognition tidak terpasang.")
            r = sr.Recognizer()
            with sr.AudioFile(audio_filepath) as source:
                audio_data = r.record(source)
            text = r.recognize_google(audio_data, language=language)
            return text

        # Perform transcription now (safe try/except)
        transcribed_text = ""
        trans_start = time.time()
        try:
            if method.startswith("Automatic (OpenAI"):
                transcribed_text = transcribe_with_openai(audio_path, language=language)
            elif method.startswith("Local Whisper"):
                transcribed_text = transcribe_local_whisper(audio_path, language=language)
            else:
                lang_map = {"auto":"en-US","id":"id-ID","en":"en-US"}
                transcribed_text = transcribe_fallback_google(audio_path, language=lang_map.get(language,"en-US"))
            trans_time = time.time() - trans_start
        except Exception as e:
            st.error(f"Transkripsi gagal: {e}")
            transcribed_text = ""
            trans_time = None

        if transcribed_text:
            st.markdown("#### Transkrip")
            st.code(transcribed_text, language="text")

            # Scoring: similarity ratio (SequenceMatcher) and WER (if jiwer present)
            ref_text = question.get("reference","")
            # Normalize simple
            def normalize_text(t):
                return " ".join(t.lower().strip().split())

            norm_ref = normalize_text(ref_text)
            norm_hyp = normalize_text(transcribed_text)

            # Similarity ratio via difflib (0..1)
            sim_ratio = SequenceMatcher(None, norm_ref, norm_hyp).ratio()
            sim_pct = sim_ratio * 100

            # WER if available
            wer_score = None
            if wer is not None and show_wer_flag and ref_text.strip() != "":
                try:
                    wer_score = wer(ref_text, transcribed_text)
                except Exception:
                    wer_score = None

            # Combined score mapping:
            # similarity has more weight (70%), WER decreases score.
            base_score = sim_pct  # 0-100
            if wer_score is not None:
                # transform WER (0 best) to penalty
                penalty = min(wer_score, 1.0) * 100  # 0-100
                final_score = max(0, base_score * 0.7 + (100 - penalty) * 0.3)
            else:
                final_score = base_score

            # Display metrics
            col_a, col_b, col_c = st.columns([1,1,1])
            col_a.metric("Similarity (%)", f"{sim_pct:.1f}%")
            col_b.metric("Final Score", f"{final_score:.1f}/100")
            if wer_score is not None:
                col_c.metric("WER", f"{wer_score:.3f}")
            else:
                col_c.markdown('<div class="small">WER: unavailable</div>', unsafe_allow_html=True)

            # Feedback text
            st.markdown("#### Feedback otomatis")
            feedback_lines = []
            if sim_pct > 85:
                feedback_lines.append("- Jawaban sangat sesuai dengan kunci. Pelafalan dan isi bagus.")
            elif sim_pct > 60:
                feedback_lines.append("- Jawaban cukup mendekati kunci. Perbaiki beberapa kosakata atau detail.")
            else:
                feedback_lines.append("- Jawaban berbeda jauh dari kunci. Pastikan menjawab poin utama soal.")
            if wer_score is not None and wer_score > 0.5:
                feedback_lines.append("- WER tinggi: audio mungkin berisik atau pelafalan tidak jelas.")
            st.write("\n".join(feedback_lines))

            # Download transcript
            st.download_button("📥 Download Transkrip (.txt)", data=transcribed_text, file_name=f"transcript_q{question['id']}.txt", mime="text/plain")

            # Show timing & meta
            st.markdown("---")
            st.markdown(f"- Metode transkripsi: **{method}**")
            if trans_time:
                st.markdown(f"- Waktu proses: **{trans_time:.1f} s**")
            st.markdown(f"- Panjang transkrip: **{len(transcribed_text.split())} kata**")

        else:
            st.warning("Transkripsi kosong atau gagal. Coba lagi dengan audio yang lebih jelas atau unggah file lain.")

    st.markdown("</div>", unsafe_allow_html=True)

# --- Footer / Extras ---
st.write("")
st.markdown('<div class="small center">Built with ❤️ — Jika mau kustom branding atau scoring lebih canggih (semantic matching), beri tahu aku.</div>', unsafe_allow_html=True)

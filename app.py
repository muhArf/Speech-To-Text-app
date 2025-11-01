# app.py
import streamlit as st
from pathlib import Path
import tempfile
import os
import io
import time
import base64

# Audio processing
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt

# Optional libs (may not be installed in target). We'll import lazily.
try:
    import openai
except Exception:
    openai = None

try:
    import whisper
except Exception:
    whisper = None

# Evaluation
try:
    from jiwer import wer
except Exception:
    wer = None

st.set_page_config(page_title="AI Assessment — Speech-to-Text", page_icon="🧠", layout="wide")

# --- Styling (embedded CSS) ---
st.markdown(
    """
    <style>
    /* page */
    .stApp { background: linear-gradient(180deg,#0f1724 0%, #071032 100%); color: #e6eef8; }
    .header { color: #ffffff; font-size:32px; font-weight:600; }
    .subheader { color: #9fb4d9; margin-top: -10px; }
    .card { background: rgba(255,255,255,0.04); padding: 18px; border-radius: 14px; box-shadow: 0 4px 20px rgba(2,6,23,0.6); }
    .muted { color: #a9bfd9; }
    .metric { background: rgba(255,255,255,0.03); padding: 12px; border-radius: 10px; }
    .small { font-size:12px; color:#9fb4d9; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Header ---
col1, col2 = st.columns([3,1])
with col1:
    st.markdown('<div class="header">AI Assessment — Speech-to-Text</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Unggah audio, dapatkan transkrip otomatis, evaluasi kualitas transkrip (WER), dan unduh hasilnya.</div>', unsafe_allow_html=True)
with col2:
    st.image("https://static.streamlit.io/examples/ai.png", width=64)  # decorative (works on Streamlit hosting)

st.write("")  # spacing

# --- Sidebar: settings ---
with st.sidebar:
    st.markdown("## Pengaturan Transkripsi")
    st.info("Pilih metode transkripsi dan bahasa.")
    method = st.selectbox("Metode", ["Automatic (OpenAI Whisper API — but needs OPENAI_API_KEY)", "Local Whisper (CPU/GPU)", "Streamlit SpeechRecognition (fallback)"])
    language = st.selectbox("Bahasa target (opsional — biarkan Auto jika unsure)", ["auto", "id", "en", "ms", "other"])
    show_wave = st.checkbox("Tampilkan waveform audio", value=True)
    enable_wer = st.checkbox("Tampilkan metrik evaluasi (WER) jika ada teks referensi", value=True)
    st.markdown("---")
    st.markdown("### Catatan deploy")
    st.markdown("- Untuk **OpenAI Whisper API**, tambahkan `OPENAI_API_KEY` pada Secrets/Environment di Streamlit Cloud.")
    st.markdown("- Untuk **Local Whisper**, instal paket `whisper` & dependencies (butuh lebih besar).")
    st.markdown("- Jika tidak punya API key, gunakan Local Whisper bila tersedia di environment.")

st.markdown("")  # spacing

# --- Upload area ---
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 1) Unggah file audio (mp3 / wav / m4a)")
uploaded = st.file_uploader("Pilih file audio", type=["wav","mp3","m4a","flac","ogg"], accept_multiple_files=False)

st.markdown("### 2) (Opsional) Unggah teks referensi untuk evaluasi (plain .txt)")
ref_file = st.file_uploader("Teks referensi (optional .txt)", type=["txt"], accept_multiple_files=False)

st.markdown("</div>", unsafe_allow_html=True)
st.write("")

# Helper: save uploaded to temp file
def save_tempfile(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tf.write(uploaded_file.getbuffer())
    tf.flush()
    tf.close()
    return Path(tf.name)

def plot_waveform(filepath, sr=16000):
    y, sr = librosa.load(filepath, sr=sr, mono=True)
    duration = len(y) / sr
    fig, ax = plt.subplots(figsize=(10,2.2))
    times = np.linspace(0, duration, num=len(y))
    ax.plot(times, y, linewidth=0.3)
    ax.set_xlim(0, duration)
    ax.set_xlabel("Time (s)")
    ax.set_yticks([])
    ax.set_title("Waveform")
    plt.tight_layout()
    return fig

def transcribe_openai(audio_path, language="auto"):
    if openai is None:
        raise RuntimeError("openai package not installed in environment.")
    key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None) if "OPENAI_API_KEY" in st.secrets else None
    if not key:
        raise RuntimeError("OPENAI_API_KEY not found in environment. Set it in Streamlit secrets or env vars.")
    openai.api_key = key
    # Use Whisper-1 via OpenAI Python package (transcribe)
    with open(audio_path, "rb") as f:
        # model "whisper-1" is used
        try:
            # This is compatible with openai-python v0.27+ style examples
            transcript = openai.Audio.transcribe("whisper-1", f, language=None if language=="auto" else language)
            # transcript is likely a dict-like object with 'text'
            text = transcript["text"] if isinstance(transcript, dict) and "text" in transcript else str(transcript)
        except Exception as e:
            raise RuntimeError(f"OpenAI transcription failed: {e}")
    return text

def transcribe_local_whisper(audio_path, language="auto"):
    if whisper is None:
        raise RuntimeError("whisper package not available in environment.")
    model_size = "small"  # safe default for Cloud; change to "base" or "tiny" for faster/cheaper
    model = whisper.load_model(model_size)
    opts = {}
    if language != "auto":
        opts["language"] = language
        opts["task"] = "transcribe"
    result = model.transcribe(str(audio_path), **opts)
    return result.get("text", "")

def transcribe_fallback_speech_recognition(audio_path, language="en-US"):
    # Basic fallback using SpeechRecognition + Google Web Speech (internet, no API key)
    try:
        import speech_recognition as sr
    except Exception:
        raise RuntimeError("speech_recognition package not installed.")
    r = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = r.record(source)
    try:
        text = r.recognize_google(audio, language=language)
    except Exception as e:
        raise RuntimeError(f"Google Speech recognition failed: {e}")
    return text

# --- Action: transcribe on button press ---
if uploaded is None:
    st.info("Unggah file audio untuk memulai transkripsi.")
else:
    # Save file
    audio_path = save_tempfile(uploaded)
    filesize_mb = os.path.getsize(audio_path) / (1024*1024)
    st.markdown(f"**Nama file:** {uploaded.name}   •   **Ukuran:** {filesize_mb:.2f} MB")
    # Playback
    st.audio(uploaded)

    # Show waveform
    if show_wave:
        try:
            fig = plot_waveform(str(audio_path))
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Gagal menampilkan waveform: {e}")

    # Transcription button
    if st.button("Transcribe sekarang", type="primary"):
        status = st.empty()
        status.info("Memulai transkripsi...")
        start = time.time()
        try:
            if method.startswith("Automatic (OpenAI"):
                status.info("Menggunakan OpenAI Whisper API...")
                transcript_text = transcribe_openai(str(audio_path), language=language)
            elif method.startswith("Local Whisper"):
                status.info("Menggunakan local Whisper model...")
                transcript_text = transcribe_local_whisper(str(audio_path), language=language)
            else:
                status.info("Menggunakan fallback SpeechRecognition...")
                # language mapping for Google recognizer
                lang_map = {"auto":"en-US","id":"id-ID","en":"en-US","ms":"ms-MY","other":"en-US"}
                transcript_text = transcribe_fallback_speech_recognition(str(audio_path), language=lang_map.get(language,"en-US"))

            duration = time.time() - start
            status.success(f"Transkripsi selesai dalam {duration:.1f} detik.")
            # Display results
            st.markdown("### Hasil Transkripsi")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.code(transcript_text, language="text")
            st.markdown("</div>", unsafe_allow_html=True)

            # Metrics / info
            with st.expander("Detail & Metrik"):
                st.markdown(f"- Model metode: **{method}**")
                st.markdown(f"- Bahasa (setting): **{language}**")
                st.markdown(f"- Waktu proses: **{duration:.1f} s**")
                st.markdown(f"- Ukuran file: **{filesize_mb:.2f} MB**")
                # Word count
                wcount = len(transcript_text.split())
                st.metric("Kata (transkrip)", f"{wcount}")

            # If reference text uploaded and jiwer available
            if ref_file is not None and enable_wer:
                try:
                    ref_text = ref_file.getvalue().decode("utf-8")
                except Exception:
                    ref_text = ref_file.getvalue().decode("latin-1", errors="ignore")
                st.markdown("### Evaluasi kinerja (WER)")
                if wer is None:
                    st.warning("Paket 'jiwer' belum terpasang — tidak dapat menghitung WER. Tambahkan 'jiwer' ke requirements.")
                else:
                    score = wer(ref_text, transcript_text)
                    st.markdown('<div class="metric">', unsafe_allow_html=True)
                    st.markdown(f"**WER (Word Error Rate):** `{score:.3f}` — semakin kecil semakin baik.")
                    # breakdown: simple substitution/insert/delete estimation not shown here (jiwer has more)
                    st.markdown("</div>", unsafe_allow_html=True)

            # Download transcript button
            bcol1, bcol2 = st.columns([1,3])
            with bcol1:
                st.download_button("📥 Download .txt", data=transcript_text, file_name=f"{Path(uploaded.name).stem}_transcript.txt", mime="text/plain")
            with bcol2:
                st.success("Transkrip siap diunduh.")

        except Exception as e:
            status.error(f"Gagal melakukan transkripsi: {e}")
            st.exception(e)

import streamlit as st
import whisperx
import torch
from sentence_transformers import SentenceTransformer, util
import tempfile
import os
import random
import subprocess
import gc

# Interview questions
INTERVIEW_QUESTIONS = [
    "Can you tell me about yourself?",
    "Why do you want to work in this company?",
    "What are your greatest strengths?",
    "What is your biggest weakness and how do you handle it?",
    "Tell me about a time you worked in a team to solve a problem.",
    "Where do you see yourself in five years?",
    "How do you handle stressful situations?",
    "Describe a challenge you faced and how you overcame it.",
    "Why should we hire you?",
    "What motivates you to do your best work?"
]

if 'question' not in st.session_state:
    st.session_state.question = random.choice(INTERVIEW_QUESTIONS)

@st.cache_resource
def load_whisperx_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)
    return model, device

@st.cache_resource
def load_sbert_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def extract_audio_from_video(video_path, audio_path):
    try:
        command = ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', 
                   '-ar', '16000', '-ac', '1', '-y', audio_path]
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except:
        return False

def transcribe_with_whisperx(audio_path, model, device):
    try:
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=16)
        
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], device=device)
        
        result = whisperx.align(result["segments"], model_a, metadata, 
                                audio, device, return_char_alignments=False)
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        del model_a
        
        return result
    except:
        return None

def format_transcript(result):
    if not result or "segments" not in result:
        return ""
    return " ".join([seg["text"].strip() for seg in result["segments"]])

def get_detailed_transcript(result):
    if not result or "segments" not in result:
        return []
    
    detailed = []
    for segment in result["segments"]:
        confidence = 0
        if "words" in segment and segment["words"]:
            confidence = sum(w.get("score", 0) for w in segment["words"]) / len(segment["words"])
        
        detailed.append({
            "start": segment.get("start", 0),
            "end": segment.get("end", 0),
            "text": segment["text"].strip(),
            "confidence": confidence
        })
    return detailed

def classify_similarity(score):
    if score >= 0.8:
        return "Sangat relevan"
    elif score >= 0.6:
        return "Cukup relevan"
    elif score >= 0.4:
        return "Kurang relevan"
    else:
        return "Tidak relevan"

def count_fillers(text):
    fillers = ["um", "uh", "hmm", "like", "you know", "okay", "erm", "ah", "well", "so"]
    return sum(text.lower().count(f) for f in fillers)

def analyze_answer(question, transcript, sbert_model):
    emb_q = sbert_model.encode(question, convert_to_tensor=True)
    emb_a = sbert_model.encode(transcript, convert_to_tensor=True)
    sim_score = util.cos_sim(emb_q, emb_a).item()
    fillers_count = count_fillers(transcript)
    
    return {
        'similarity_score': sim_score,
        'category': classify_similarity(sim_score),
        'fillers_count': fillers_count,
        'word_count': len(transcript.split()),
        'smooth_speech': fillers_count <= 3
    }

st.title("AI Video Interview Assessment")
st.write("Upload your video interview for AI analysis")

device_info = "GPU Available" if torch.cuda.is_available() else "Using CPU"
st.info(f"System: {device_info}")

with st.spinner("Loading models..."):
    whisperx_model, device = load_whisperx_model()
    sbert_model = load_sbert_model()

st.success("Models loaded")

st.subheader("Interview Question:")
st.info(st.session_state.question)

if st.button("Generate New Question"):
    st.session_state.question = random.choice(INTERVIEW_QUESTIONS)
    st.rerun()

st.subheader("Upload Video")
video_file = st.file_uploader("Select video file", type=['mp4', 'avi', 'mov', 'mkv', 'webm'])

if video_file:
    st.video(video_file)
    st.write(f"File size: {video_file.size / (1024 * 1024):.2f} MB")
    
    if st.button("Process Video"):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
            tmp_video.write(video_file.read())
            video_path = tmp_video.name
        
        audio_path = video_path.replace('.mp4', '.wav')
        
        try:
            with st.spinner("Extracting audio..."):
                if not extract_audio_from_video(video_path, audio_path):
                    st.error("Audio extraction failed")
                    st.stop()
                st.success("Audio extracted")
            
            with st.spinner("Transcribing..."):
                result = transcribe_with_whisperx(audio_path, whisperx_model, device)
                if not result:
                    st.error("Transcription failed")
                    st.stop()
            
            transcript = format_transcript(result)
            detailed_segments = get_detailed_transcript(result)
            
            if not transcript:
                st.warning("No speech detected")
                st.stop()
            
            st.subheader("Transcript:")
            st.text_area("Full Text", transcript, height=150)
            
            with st.expander("Detailed Transcript"):
                for seg in detailed_segments:
                    st.write(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] (Confidence: {seg['confidence']:.2%})")
                    st.write(seg['text'])
                    st.divider()
            
            with st.spinner("Analyzing..."):
                analysis = analyze_answer(st.session_state.question, transcript, sbert_model)
            
            st.subheader("Analysis Results:")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Similarity", f"{analysis['similarity_score']:.2f}")
            col2.metric("Relevance", analysis['category'])
            col3.metric("Fillers", analysis['fillers_count'])
            col4.metric("Words", analysis['word_count'])
            
            st.subheader("Speech Quality:")
            if analysis['smooth_speech']:
                st.success("Smooth speech - Good delivery")
            else:
                st.warning(f"{analysis['fillers_count']} filler words detected")
            
            st.subheader("Feedback:")
            if analysis['similarity_score'] >= 0.8:
                st.success("Excellent answer - Very relevant")
            elif analysis['similarity_score'] >= 0.6:
                st.info("Good answer - Mostly relevant")
            elif analysis['similarity_score'] >= 0.4:
                st.warning("Answer could be more focused")
            else:
                st.error("Answer seems off-topic")
            
            if result.get("language"):
                st.info(f"Language: {result['language'].upper()}")
        
        finally:
            if os.path.exists(video_path):
                os.unlink(video_path)
            if os.path.exists(audio_path):
                os.unlink(audio_path)

st.markdown("---")
st.markdown("Built with WhisperX & SentenceTransformers")
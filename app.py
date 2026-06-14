import streamlit as st
import torch
from transformers import pipeline
import moviepy.editor as mp
import os
import tempfile
import gc

st.set_page_config(page_title="Universal Media Transcriber", layout="wide")
st.title("🎙️ Audio & Video Transcription Assistant")

@st.cache_resource
def load_transcriber():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Fallback to 'whisper-base' if deploying on 1GB RAM limits
    return pipeline(
        "automatic-speech-recognition", 
        model="openai/whisper-base", 
        device=device,
        chunk_length_s=30
    )

transcriber = load_transcriber()

SUPPORTED_FORMATS = ["mp3", "wav", "mp4", "ts", "mov", "mkv", "avi"]
uploaded_file = st.file_uploader("Select Media File", type=SUPPORTED_FORMATS)

if uploaded_file is not None:
    file_extension = os.path.splitext(uploaded_file.name)[1]
    
    # 1. Write and CLOSE the file to prevent OS permission locks
    tmp_media = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
    tmp_media.write(uploaded_file.read())
    tmp_media_path = tmp_media.name
    tmp_media.close() 

    st.info("File uploaded successfully. Processing media...")
    tmp_audio_path = "temp_audio_processing.wav"

    try:
        with st.spinner("Extracting audio track..."):
            # Extract audio
            clip = mp.AudioFileClip(tmp_media_path)
            clip.write_audiofile(tmp_audio_path, logger=None)
            
            # Explicitly close the clip to free up RAM before model inference
            clip.close()
            
        with st.spinner("Generating transcript (this may take a moment on CPU)..."):
            # Batch size reduced to 2 to prevent RAM spikes on cloud instances
            result = transcriber(tmp_audio_path, batch_size=2)
            transcript_text = result["text"]
            
            st.success("Transcription Complete!")
            st.text_area("Transcript", transcript_text, height=300)
            
            st.download_button(
                label="📥 Download Transcript as TXT",
                data=transcript_text,
                file_name="transcript.txt",
                mime="text/plain"
            )
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        
    finally:
        # 2. Aggressive cleanup and forced garbage collection
        if os.path.exists(tmp_media_path):
            os.remove(tmp_media_path)
        if os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)
        gc.collect()

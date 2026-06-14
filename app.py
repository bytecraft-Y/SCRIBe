import streamlit as st
import torch
from transformers import pipeline
import moviepy.editor as mp
import os
import tempfile
import gc

st.set_page_config(page_title="Universal Media Transcriber", layout="wide")
st.title("🎙️ Audio & Video Transcription Assistant")
st.write("Optimized with Whisper-Tiny for fast, ultra-light cloud execution.")

@st.cache_resource
def load_transcriber():
    # Force CPU usage if GPU isn't available, which is standard for free Streamlit Cloud
    device = 0 if torch.cuda.is_available() else -1
    
    # 'openai/whisper-tiny' is ultra-lightweight (~75MB file size, ~150MB operational RAM)
    return pipeline(
        "automatic-speech-recognition", 
        model="openai/whisper-tiny", 
        device=device,
        chunk_length_s=30
    )

transcriber = load_transcriber()

SUPPORTED_FORMATS = ["mp3", "wav", "mp4", "ts", "mov", "mkv", "avi"]
uploaded_file = st.file_uploader("Select Media File", type=SUPPORTED_FORMATS)

if uploaded_file is not None:
    file_extension = os.path.splitext(uploaded_file.name)[1]
    
    # Secure binary write and immediate file closing to handle cross-OS file locking
    tmp_media = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
    tmp_media.write(uploaded_file.read())
    tmp_media_path = tmp_media.name
    tmp_media.close() 

    st.info("File uploaded successfully. Processing media...")
    tmp_audio_path = "temp_audio_processing.wav"

    try:
        with st.spinner("Extracting audio track..."):
            clip = mp.AudioFileClip(tmp_media_path)
            clip.write_audiofile(tmp_audio_path, logger=None)
            clip.close()  # Clear video file from memory immediately
            
        with st.spinner("Generating transcript..."):
            # batch_size=1 ensures the CPU processes one chunk at a time, minimizing RAM spikes
            result = transcriber(tmp_audio_path, batch_size=1)
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
        # Aggressive cleanup of local file system buffers
        if os.path.exists(tmp_media_path):
            os.remove(tmp_media_path)
        if os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)
        gc.collect()  # Explicitly trigger Python garbage collection

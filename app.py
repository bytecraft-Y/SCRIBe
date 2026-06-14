import streamlit as st
import torch
from transformers import pipeline
import moviepy.editor as mp
import os
import tempfile

# Configure the Streamlit UI
st.set_page_config(page_title="Universal Media Transcriber", layout="wide")
st.title("🎙️ Audio & Video Transcription Assistant")
st.write("Upload media files (MP3, WAV, MP4, MKV, MOV, TS, AVI) to generate a text transcript.")

# Cache the model so it only downloads/loads into memory once
@st.cache_resource
def load_transcriber():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Using 'whisper-base' or 'whisper-small' is recommended for Streamlit Cloud memory limits
    return pipeline(
        "automatic-speech-recognition", 
        model="openai/whisper-small", 
        device=device,
        chunk_length_s=30
    )

transcriber = load_transcriber()

# Supported formats based on FFmpeg capabilities
SUPPORTED_FORMATS = ["mp3", "wav", "mp4", "ts", "mov", "mkv", "avi"]
uploaded_file = st.file_uploader("Select Media File", type=SUPPORTED_FORMATS)

if uploaded_file is not None:
    # Safely handle the uploaded file via a temporary directory
    file_extension = os.path.splitext(uploaded_file.name)[1]
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_media:
        tmp_media.write(uploaded_file.read())
        tmp_media_path = tmp_media.name

    st.info("File uploaded successfully. Processing media...")

    try:
        with st.spinner("Extracting audio track and generating transcript..."):
            # Extract audio to a temporary WAV file
            tmp_audio_path = "temp_audio_processing.wav"
            clip = mp.AudioFileClip(tmp_media_path)
            clip.write_audiofile(tmp_audio_path, logger=None)
            
            # Execute transcription
            result = transcriber(tmp_audio_path, batch_size=8)
            transcript_text = result["text"]
            
            # Display Results
            st.success("Transcription Complete!")
            st.text_area("Transcript", transcript_text, height=300)
            
            # Download capability
            st.download_button(
                label="📥 Download Transcript as TXT",
                data=transcript_text,
                file_name="transcript.txt",
                mime="text/plain"
            )
            
    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
        
    finally:
        # Strict garbage collection to prevent storage leaks on the server
        if os.path.exists(tmp_media_path):
            os.remove(tmp_media_path)
        if os.path.exists("temp_audio_processing.wav"):
            os.remove("temp_audio_processing.wav")
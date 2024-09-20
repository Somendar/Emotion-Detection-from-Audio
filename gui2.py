import streamlit as st
from streamlit_webrtc import webrtc_streamer, ClientSettings
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import librosa
import tempfile
import os
import queue
import av
from typing import List

# Set page configuration before any other code
st.set_page_config(page_title="Emotion Detection", page_icon="🎤", layout="centered")

# Load emotion classification model
@st.cache_resource
def load_emotion_model():
    model = load_model('Emodel.h5')
    return model

emotion_model = load_emotion_model()

# Emotion labels mapping
emotion_labels = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'}


# Check model input shape
model_input_shape = emotion_model.input_shape
print(f"Model input shape: {model_input_shape}")

# Gender detection function
def detect_gender(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = pitches[np.nonzero(pitches)]
    pitch_mean = np.mean(pitch)
    
    if pitch_mean > 160:
        return 'Female'
    else:
        return 'Male'

# Audio preprocessing function
def preprocess_audio(audio_path, model):
    y, sr = librosa.load(audio_path, sr=22050)
    
    if len(y) < 3 * sr:
        padding = 3 * sr - len(y)
        y = np.pad(y, (0, padding), 'constant')
    else:
        y = y[:3 * sr]
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    
    # Ensure the shape matches what the model expects
    model_input_shape = model.input_shape[1:]  # Excluding batch dimension
    if len(model_input_shape) == 2:  # Expecting (time_steps, features)
        model_input = np.expand_dims(mfccs_processed, axis=0)  # Shape should be (1, features)
        model_input = np.expand_dims(model_input, axis=2)     # Shape should be (1, features, 1) if model uses 1D Conv
    elif len(model_input_shape) == 3:  # Expecting (batch, time_steps, features)
        # Ensure the MFCCs match the time steps and features expected
        num_features = model_input_shape[1]
        num_time_steps = model_input_shape[0]
        if len(mfccs_processed) < num_features:
            padding = num_features - len(mfccs_processed)
            mfccs_processed = np.pad(mfccs_processed, (0, padding), 'constant')
        else:
            mfccs_processed = mfccs_processed[:num_features]
        model_input = np.expand_dims(mfccs_processed, axis=0)  # Shape should be (1, num_features)
        model_input = np.expand_dims(model_input, axis=2)     # Shape should be (1, num_features, 1) if model uses 1D Conv
    else:
        raise ValueError(f"Unsupported input shape configuration: {model_input_shape}")

    return model_input

# Enhanced Streamlit App Layout

# Title and Description
st.title("🎤 Emotion Detection from Female Voice Recordings")

st.markdown("""
Welcome to the **Emotion Detection App**! This application uses a pre-trained model to detect emotions from female voice recordings.

- 🎤 **Record** your voice directly in the app.
- 📤 **Upload** an audio file (.wav, .mp3, or .ogg).

Please note: This model is trained to detect emotions **only for female voices**.
""")

# Separator Line
st.markdown("---")

# Sidebar with instructions
with st.sidebar:
    st.header("Instructions")
    st.write("""
    1. **Record** or **upload** your voice.
    2. Ensure the voice is female for best results.
    3. The app will detect the emotion and display it on the screen.
    """)

# Emotion Detection Section
st.subheader("Detect Emotion from Your Voice")

# Audio Recording Functionality
st.markdown("### Record Your Voice")

# Audio Recording Block
if st.button("🎙 Start Recording"):
    st.info("Recording... Please speak into your microphone.")
    
    audio_q = queue.Queue()

    def audio_callback(frame):
        audio_q.put(frame)
        return av.AudioFrame()

    webrtc_ctx = webrtc_streamer(
        key="speech-recognizer",
        mode="sendonly",
        audio_receiver_size=256,
        audio_frame_callback=audio_callback,
        client_settings=ClientSettings(
            media_stream_constraints={"audio": True, "video": False},
        ),
    )

    if webrtc_ctx.state.playing:
        st.success("Recording complete. Processing...")

        # Collect audio frames from recording
        audio_frames: List[av.AudioFrame] = []
        while True:
            try:
                frame = audio_q.get(timeout=1.0)
                audio_frames.append(frame)
            except queue.Empty:
                break

        if audio_frames:
            # Save audio to temp file
            audio = b''.join([frame.to_ndarray().tobytes() for frame in audio_frames])
            temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            with open(temp_audio_file.name, 'wb') as f:
                f.write(audio)

            st.audio(temp_audio_file.name, format='audio/wav')

            gender = detect_gender(temp_audio_file.name)
            if gender != 'Female':
                st.warning("Detected voice is not female. Please record a female voice.")
            else:
                processed_audio = preprocess_audio(temp_audio_file.name, emotion_model)
                prediction = emotion_model.predict(processed_audio)
                predicted_emotion = emotion_labels[np.argmax(prediction)]
                st.success(f"**Predicted Emotion:** {predicted_emotion}")

            os.unlink(temp_audio_file.name)
        else:
            st.error("No audio recorded. Please try again.")

# Separator Line
st.markdown("---")

# Audio Upload Functionality
st.markdown("### Upload an Audio File")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_audio_file.write(uploaded_file.read())
    temp_audio_file.close()

    gender = detect_gender(temp_audio_file.name)
    if gender != 'Female':
        st.warning("Detected voice is not female. Please upload a female voice recording.")
    else:
        processed_audio = preprocess_audio(temp_audio_file.name, emotion_model)
        prediction = emotion_model.predict(processed_audio)
        predicted_emotion = emotion_labels[np.argmax(prediction)]
        st.success(f"**Predicted Emotion:** {predicted_emotion}")

    os.unlink(temp_audio_file.name)

# Footer
st.markdown("---")
st.markdown("Developed by [Your Name](https://github.com/YourProfile)")

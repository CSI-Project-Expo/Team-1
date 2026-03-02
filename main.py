import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import dlib
from deepface import DeepFace
from collections import Counter
from PIL import Image
from moviepy import VideoFileClip
from pydub import AudioSegment
import speech_recognition as sr
import time

# -----------------------------
# CUSTOM STYLING
# -----------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #1f1c2c, #928dab);
    color: white;
}

h1 {
    text-align: center;
    color: #ff4b4b;
    font-size: 42px;
    font-weight: bold;
    text-shadow: 2px 2px 8px black;
}

h2, h3 {
    color: #ffcc70;
}

.stRadio > div {
    background-color: rgba(255,255,255,0.1);
    padding: 10px;
    border-radius: 10px;
}

.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton>button:hover {
    background-color: #ff1e1e;
    transform: scale(1.05);
}

.stFileUploader {
    background-color: rgba(255,255,255,0.1);
    padding: 10px;
    border-radius: 10px;
}


/* SUCCESS MESSAGE (Safe) */
div[data-testid="stSuccess"] {
    background-color: #00ff99 !important;
    color: black !important;
    font-weight: bold !important;
    border-radius: 12px !important;
    padding: 15px !important;
    font-size: 16px !important;
}

/* ERROR MESSAGE (Unsafe) */
div[data-testid="stError"] {
    background-color: #ff4b4b !important;
    color: white !important;
    font-weight: bold !important;
    border-radius: 12px !important;
    padding: 15px !important;
    font-size: 16px !important;
}

/* File uploader button */
.stFileUploader button {
    background-color: #00c6ff !important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: bold !important;
}

/* Hover effect */
.stFileUploader button:hover {
    background-color: #0072ff !important;
    color: white !important;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# CONFIG
# -----------------------------
HARSH_WORDS = ["leave", "help", "weird", "danger", "stop", "attack", "scared", "kill"]
SHOUTING_THRESHOLD_DBFS = -15.0
UNSAFE_EMOTIONS = {'angry', 'sad', 'fear', 'disgust'}

# -----------------------------
# LOAD DLIB
# -----------------------------
@st.cache_resource
def load_models():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    return detector, predictor

detector, predictor = load_models()

# -----------------------------
# FACIAL MOVEMENT
# -----------------------------
def get_facial_movement_score(landmarks):
    points = landmarks.parts()
    return abs(points[57].y - points[51].y)

# -----------------------------
# BODY MOVEMENT
# -----------------------------
def detect_body_movement(prev_frame, current_frame):

    if prev_frame is None:
        return False, current_frame

    gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray1, gray2)
    blur = cv2.GaussianBlur(diff, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)

    movement_score = np.sum(thresh)

    if movement_score > 5000000:
        cv2.putText(current_frame, "AGGRESSIVE BODY MOVEMENT!",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2)
        return True, current_frame

    return False, current_frame

# -----------------------------
# VISUAL ANALYSIS
# -----------------------------
def analyze_visuals(frame, prev_frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    emotions_detected = []
    danger_detected = False

    results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

    for res in results:

        dom_emotion = res['dominant_emotion']
        emotions_detected.append(dom_emotion)

        region = res['region']
        x, y, w, h = region['x'], region['y'], region['w'], region['h']

        box_color = (0, 255, 0)
        label = dom_emotion.capitalize()

        try:
            rect = dlib.rectangle(x, y, x + w, y + h)
            landmarks = predictor(gray, rect)
            movement = get_facial_movement_score(landmarks)

            if movement > 79:
                box_color = (0, 0, 255)
                label = "DANGEROUS FACIAL MOVEMENT!"
                danger_detected = True

        except:
            pass

        if dom_emotion in UNSAFE_EMOTIONS:
            box_color = (0, 0, 255)
            label = f"ALERT: {dom_emotion.capitalize()}"
            danger_detected = True

        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
        cv2.putText(frame, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    body_danger, frame = detect_body_movement(prev_frame, frame)

    if body_danger:
        danger_detected = True

    return frame, emotions_detected, danger_detected

# -----------------------------
# AUDIO ANALYSIS
# -----------------------------
def analyze_audio(file_path):

    recognizer = sr.Recognizer()
    text = ""
    found_words = []
    shouting = False
    temp_wav = "temp_audio.wav"

    try:
        if file_path.endswith((".mp4", ".avi", ".mov", ".mkv")):
            video = VideoFileClip(file_path)
            if video.audio is None:
                return None, [], False
            video.audio.write_audiofile(temp_wav, codec='pcm_s16le', logger=None)
            audio_segment = AudioSegment.from_file(temp_wav)
        else:
            audio_segment = AudioSegment.from_file(file_path)
            audio_segment.export(temp_wav, format="wav")

        if audio_segment:
            shouting = audio_segment.dBFS > SHOUTING_THRESHOLD_DBFS

        with sr.AudioFile(temp_wav) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data).lower()
            found_words = [w for w in HARSH_WORDS if w in text]

        return text, found_words, shouting

    except:
        return None, [], False

    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)

# -----------------------------
# TITLE (Styled)
# -----------------------------
st.markdown("<h1>🚨 AI Women Safety Monitoring System</h1>", unsafe_allow_html=True)
st.markdown("<center><h3>Real-Time Emotion • Audio • Threat Detection</h3></center>", unsafe_allow_html=True)

mode = st.radio("Choose Mode:", ["Image", "Video", "Audio","Live Monitoring"])

# (Rest of your code continues EXACTLY SAME — I did not change any logic below)
# IMAGE MODE
if mode == "Image":

    uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded:
        img = Image.open(uploaded)
        frame = np.array(img)

        processed, emotions, danger = analyze_visuals(frame, None)

        st.image(processed, channels="BGR")

        if danger:
            st.error("🚨 Unsafe Situation Detected!")
        else:
            st.success("✅ Situation Appears Safe.")


# VIDEO MODE
elif mode == "Video":

    uploaded = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded:

        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded.read())
        video_path = temp_file.name

        cap = cv2.VideoCapture(video_path)

        stframe = st.empty()
        emotion_counter = Counter()
        danger_count = 0
        frame_count = 0
        prev_frame = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % 5 == 0:
                processed, emotions, danger = analyze_visuals(frame, prev_frame)

                prev_frame = frame.copy()

                for e in emotions:
                    emotion_counter[e] += 1

                if danger:
                    danger_count += 1

                stframe.image(processed, channels="BGR")

        cap.release()

        text, words, shouting = analyze_audio(video_path)

        st.subheader("📋 Final Report")

        st.write("Emotion Distribution:", dict(emotion_counter))
        st.write("Danger Frames:", danger_count)

        if words:
            st.error(f"⚠️ Harsh Words: {', '.join(words)}")

        if shouting:
            st.error("⚠️ Shouting Detected!")

        if (any(e in UNSAFE_EMOTIONS for e in emotion_counter)
            or danger_count > 5
            or words
            or shouting):
            st.error("🚨 FINAL RESULT: Unsafe Situation Detected!")
        else:
            st.success("✅ FINAL RESULT: Situation Appears Safe")


# AUDIO MODE
elif mode == "Audio":

    uploaded = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])

    if uploaded:

        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded.read())
        audio_path = temp_file.name

        text, words, shouting = analyze_audio(audio_path)

        if words:
            st.error(f"⚠️ Harsh Words: {', '.join(words)}")

        if shouting:
            st.error("⚠️ Shouting Detected!")

        if not words and not shouting:
            st.success("✅ Audio Appears Safe.")
# -----------------------------
# LIVE WEBCAM MONITORING
# -----------------------------
elif mode == "Live Monitoring":

    st.subheader("🔴 Real-Time Webcam Monitoring")

    start = st.button("Start Monitoring")
    stop = st.button("Stop Monitoring")
    panic = st.button("🚨 PANIC ALERT")
    call = st.button("📞 Emergency Call")

    if panic:
        st.error("🚨 PANIC ALERT ACTIVATED!")
        st.audio("https://www.soundjay.com/buttons/beep-07.wav")

    if call:
        st.error("📞 Calling Emergency Contact...")
        st.write("Dialing: 100 (Police)")

    if start:

        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        prev_frame = None
        danger_counter = 0

        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
                break

            processed, emotions, danger = analyze_visuals(frame, prev_frame)
            prev_frame = frame.copy()

            if danger:
                danger_counter += 1

            if danger_counter > 3:
                cv2.putText(processed, "🚨 HIGH RISK DETECTED!",
                            (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            3)

                st.audio("https://www.soundjay.com/buttons/beep-07.wav")

            stframe.image(processed, channels="BGR")

            if stop:
                break

        cap.release()
        st.success("Monitoring Stopped.")




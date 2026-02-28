Install the required Libraries

    %pip install opencv-python deepface tf-keras
    %pip install opencv-python dlib numpy
    !pip install SpeechRecognition pydub moviepy
Import the Libraries

    import os
    import cv2
    import dlib
    from deepface import DeepFace
    from collections import Counter
    import speech_recognition as sr
    from moviepy.editor import VideoFileClip
    from pydub import AudioSegment
    from google.colab.patches import cv2_imshow

    if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
        print("Downloading dlib model...")
        !wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        !bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    HARSH_WORDS = ["leave", "help", "weired", "danger", "stop", "attack", "scared", "kill"]

--- VISUAL PROCESSING FUNCTIONS ---
Calculates a movement score based on mouth opening distance

    def get_facial_movement_score(landmarks):
        points = landmarks.parts()
        mouth_top = points[51].y
        mouth_bottom = points[57].y
        return abs(mouth_bottom - mouth_top)
Analyzes a single frame for emotions, movements, and masks

    def analyze_visuals(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        unsafe_emotions = {'angry', 'sad', 'fear', 'disgust'}
        emotions_in_frame = []

        try:
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

            for i, res in enumerate(results):
                dom_emotion = res['dominant_emotion']
                emotion_confidence = res['emotion'][dom_emotion]
                emotions_in_frame.append(dom_emotion)

                region = res['region']
                x, y, w, h = region['x'], region['y'], region['w'], region['h']

            # Logic for Visual Alerts
                box_color = (0, 255, 0) # Default Green
                status_text = ""
                is_dangerous_movement = False

            # Check Movement
                try:
                    dlib_rect = dlib.rectangle(x, y, x + w, y + h)
                    landmarks = predictor(gray, dlib_rect)
                    movement_score = get_facial_movement_score(landmarks)
                    if movement_score > 65:
                        box_color = (0, 0, 255) # Red
                        status_text = "DANGEROUS MOVEMENT!"
                        is_dangerous_movement = True
                except:
                    pass

            # Check Emotions/Masks if no movement alert
                if not is_dangerous_movement:
                    if dom_emotion in unsafe_emotions:
                        box_color = (0, 0, 255)
                        status_text = f"ALERT: {dom_emotion.capitalize()}"
                    elif dom_emotion == 'neutral':
                        box_color = (0, 255, 255) # Yellow
                        status_text = f"Neutral ({emotion_confidence:.2f}%)"
                    elif emotion_confidence < 45:
                        box_color = (230, 216, 173) # Beige
                        status_text = "Mask/Occlusion"
                    else:
                        status_text = f"{dom_emotion.capitalize()}"

                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                cv2.putText(frame, status_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            # Add conditional print statements
                if box_color == (0, 0, 255) or box_color == (0, 255, 255):
                    print(f"   Alert triggered: {status_text}")
                elif box_color == (0, 255, 0):
                    print("You are Safe! Keep going")
  
        # Group Analysis
            if len(emotions_in_frame) > 1:
                counts = Counter(emotions_in_frame)
                for emo, count in counts.items():
                    if count > 1 and emo in unsafe_emotions:
                        print(f"🚨 GROUP ALERT: {count} people showing '{emo}'!")

        except Exception:
            pass

        return frame

--- AUDIO PROCESSING FUNCTIONS ---
    Extracts text and checks for harsh words in audio/video files

    def process_audio(file_path):
        recognizer = sr.Recognizer()
        temp_wav = "temp_audio_conversion.wav"

        try:
            if file_path.lower().endswith(('.mp4', '.mkv', '.mov', '.avi')):
                video = VideoFileClip(file_path)
                if video.audio is None: return None, "NO_AUDIO_TRACK_FOUND"
                video.audio.write_audiofile(temp_wav, codec='pcm_s16le', verbose=False, logger=None)
                audio_target = temp_wav
            else:
                audio = AudioSegment.from_file(file_path)
                audio.export(temp_wav, format="wav")
                audio_target = temp_wav

            with sr.AudioFile(audio_target) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data).lower()

            found_words = [word for word in HARSH_WORDS if word in text]
            return text, found_words

        except Exception as e:
            return None, str(e)
        finally:
            if os.path.exists(temp_wav): os.remove(temp_wav)

--- MAIN EXECUTION ---
    Main function to process visual and audio safety in a dataset

    def run_combined_safety_analysis(folder_path):
        if not os.path.exists(folder_path):
            print("❌ Folder not found.")
            return

        image_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        video_exts = ('.mp4', '.avi', '.mov', '.mkv')
        audio_only_exts = ('.wav', '.mp3', '.m4a')

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            ext = os.path.splitext(filename)[1].lower()
            print(f"\n🔍 ANALYZING: {filename}")

1. Visual Analysis (Images and Videos)
   
            if ext in image_exts:
                img = cv2.imread(file_path)
                if img is not None:
                    processed_img = analyze_visuals(img)
                    cv2_imshow(processed_img) # Uncommented

            elif ext in video_exts:
                text, words = process_audio(file_path)
                if words: print(f"   ⚠️ AUDIO ALERT! Harsh words found: {', '.join(words)}")

                cap = cv2.VideoCapture(file_path)
                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    if frame_count % 30 == 0: # Increased interval to 30 for performance
                        processed_frame = analyze_visuals(frame)
                        cv2_imshow(processed_frame) # Uncommented
                    frame_count += 1
                cap.release()

2. Audio-Only Analysis
   
            elif ext in audio_only_exts:
                text, words = process_audio(file_path)
                if text:
                    status = f"⚠️ ALERT: {', '.join(words)}" if words else "✅ CLEAN"
                    print(f"   [{status}] Transcription: {text}")

        print("\n✅ Dataset Analysis Complete.")

Run the analysis

    run_combined_safety_analysis('/content/drive/MyDrive/Women Safety/train/fear')

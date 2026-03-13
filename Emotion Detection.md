#    Facial Emotion and Body Movement Detection

--Upgrade pip first--

    !pip install --upgrade pip

--instal deepface and dlib libraries--

    %pip install opencv-python deepface tf-keras
    %pip install opencv-python dlib numpy

--import the required libraries--

    import cv2
    import dlib
    import os
    import numpy as np
    from deepface import DeepFace
    from collections import Counter
    from google.colab.patches import cv2_imshow



--Load Dlib's face detector and the facial landmark predictor once for efficiency--

    if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
        print("Downloading dlib model...")
        !wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        !bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def get_facial_movement_score(landmarks):
    
---Calculates a movement score based on mouth opening distance---
    
        points = landmarks.parts()
        mouth_top = points[51].y
        mouth_bottom = points[57].y
        return abs(mouth_bottom - mouth_top)

    def analyze_and_process(frame):
    
---Main function merging emotion analysis and movement detection for multiple people in a single frame---
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        unsafe_emotions = {'angry', 'sad', 'fear', 'disgust'}
        emotions_in_frame = []

        try:
1. Analyze Emotions using DeepFace
---enforce_detection=False allows the code to continue even if no face is clear---
        
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

            for i, res in enumerate(results):
                dom_emotion = res['dominant_emotion']
                emotion_confidence = res['emotion'][dom_emotion] # Get confidence score
                emotions_in_frame.append(dom_emotion)

--Extract coordinates for the bounding box--
            
                region = res['region']
                x, y, w, h = region['x'], region['y'], region['w'], region['h']

--Default state: Normal (Green)--

                box_color = (0, 255, 0)
                status_text = "" # Initialize status_text to be filled by conditions

2. Analyze Movements using Dlib

                dlib_rect = dlib.rectangle(x, y, x + w, y + h)
                is_dangerous_movement = False # Flag to prioritize Dlib alert
                try:
                    landmarks = predictor(gray, dlib_rect)
                    movement_score = get_facial_movement_score(landmarks)

   --Check for Unusual Movement Alert--
   
                    if movement_score > 30: # Adjust this threshold based on resolution/need
                        box_color = (0, 0, 255) # Red for Danger
                        status_text = "DANGEROUS MOVEMENT!"
                        print(f"!!! ALERT: Unusual movement detected on Person {i+1} !!!")
                        is_dangerous_movement = True
                except:
                    pass

3. Check for Unsafe Emotion, Neutral, or Mask Alerts if no dangerous movement

                if not is_dangerous_movement:
                    if dom_emotion in unsafe_emotions:
                        box_color = (0, 0, 255) # Red
                        status_text = f"ALERT: {dom_emotion.upper()} ({emotion_confidence:.2f}%)"
                        print(f"⚠️ ALERT: {dom_emotion} emotion ({emotion_confidence:.2f}%) on Person {i+1}")
                    elif dom_emotion == 'neutral': # Neutral emotion - now explicitly handled
                        box_color = (0, 255, 255) # Yellow
                        status_text = f"Neutral ({emotion_confidence:.2f}%)"
                        print(f"⚠️ ALERT: Neutral emotion ({emotion_confidence:.2f}%) on Person {i+1}")
                    elif emotion_confidence < 45: # Mask/Occlusion detection based on low confidence
                        box_color = (230,216,173) # Beige/Light Brown
                        status_text = f"Mask/Occlusion ({emotion_confidence:.2f}%)"
                        print(f"😷 ALERT: Mask/Occlusion detected ({emotion_confidence:.2f}%) on Person {i+1}")
                    else: # Default for normal emotions (happy, surprise)
                        box_color = (0, 255, 0) # Green
                        status_text = f"{dom_emotion.capitalize()} ({emotion_confidence:.2f}%)"
   
                    # No console print for normal emotions to avoid clutter
   

   --Draw Visuals--
   
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                cv2.putText(frame, status_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2) # Adjusted y-coordinate and font scale

5. Group Analysis

            if len(emotions_in_frame) > 1:
                counts = Counter(emotions_in_frame)
                for emo, count in counts.items():
                    if count > 1 and emo in unsafe_emotions:
                        print(f"🚨 GROUP ALERT: {count} people are showing '{emo}' simultaneously!")

        except Exception as e:
--Occurs if DeepFace cannot process the frame correctly--

            pass

        return frame

    def run_safety_analysis(folder_path):
--Processes all images and videos in the specified directory--

        image_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        video_exts = ('.mp4', '.avi', '.mov', '.mkv')

        if not os.path.exists(folder_path):
            print(f"Error: Folder '{folder_path}' not found.")
            return

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            ext = os.path.splitext(filename)[1].lower()

            if ext in image_exts:
                print(f"\nProcessing Image: {filename}")
                img = cv2.imread(file_path)
                if img is not None:
                    processed_img = analyze_and_process(img)
                    cv2_imshow(processed_img)

            elif ext in video_exts:
                print(f"\nProcessing Video: {filename}")
                cap = cv2.VideoCapture(file_path)
                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break

--Process every 10th frame to speed up video analysis in Colab--

                    if frame_count % 10 == 0:
                        processed_frame = analyze_and_process(frame)
                        cv2_imshow(processed_frame)
                    frame_count += 1
                cap.release()

--- EXECUTION ---
Replace this with the path to your dataset folder

    run_safety_analysis('/content/dataset')

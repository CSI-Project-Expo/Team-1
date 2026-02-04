#    Facial Emotion Detection

-- install the DeepFace Library--

    %pip install opencv-python deepface tf-keras

--import the required libraries--

    import cv2
    import os
    import numpy as np
    from deepface import DeepFace
    from google.colab.patches import cv2_imshow
    from collections import Counter

"""
Analyzes a single frame for emotions and draws colored bounding boxes.
"""

        def analyze_frame(frame):
        try:
DeepFace.analyze returns a list of dicts (one per face)

            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        
            unsafe_emotions = {'angry', 'sad', 'fear', 'disgust'}
            emotions_in_frame = []

            for i, res in enumerate(results):
                dom_emotion = res['dominant_emotion']
                emotions_in_frame.append(dom_emotion)

--- Feature 1 & 2: Bounding Boxes ---
Extract face coordinates (x, y, w, h)

                region = res['region']
                x, y, w, h = region['x'], region['y'], region['w'], region['h']
            
Default color (Green for Happy/Surprise/Other)

                box_color = (0, 255, 0) 
            
Set Red for Unsafe Emotions

                if dom_emotion in unsafe_emotions:
                    box_color = (0, 0, 255) # BGR for Red
                    print(f"⚠️ ALERT: Unsafe Emotion ({dom_emotion}) on Person {i+1}")
            
Set Yellow for Neutral Emotion

                elif dom_emotion == 'neutral':
                    box_color = (0, 255, 255) # BGR for Yellow
                    print(f"⚠️ ALERT: Neutral Emotion ({dom_emotion}) on Person {i+1}")

                else:
                    print(f"The person {i+1} is normal")
            
Draw the rectangle on the frame

                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 3)
            
Optional: Add text label above the box

                cv2.putText(frame, dom_emotion.upper(), (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

--- Mask Detection Logic ---

                emotion_confidence = res['emotion'][dom_emotion]
                if emotion_confidence < 45:
                    print(f"😷 ALERT: Mask/Occlusion detected on Person {i+1}")

--- Group Synchrony Check ---

            if len(emotions_in_frame) > 1:
                counts = Counter(emotions_in_frame)
                for emo, count in counts.items():
                    if count > 1:
                        print(f"🚨 GROUP ALERT: {count} people are showing '{emo}'")
                    else:
                    print("Keep going! People are probably normal.")

            return len(results), emotions_in_frame

        except Exception as e:
        return 0, []

    def process_dataset(directory_path):
        image_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        video_exts = ('.mp4', '.avi', '.mov', '.mkv')

        if not os.path.exists(directory_path):
            print(f"Error: Path '{directory_path}' not found.")
            return

        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            ext = os.path.splitext(filename)[1].lower()

            if ext in image_exts:
                print(f"\n--- 📷 Processing Image: {filename} ---")
                frame = cv2.imread(file_path)
                if frame is not None:
                    analyze_frame(frame)
                    cv2_imshow(frame) # Now displays with colored boxes

            elif ext in video_exts:
                print(f"\n--- 🎥 Processing Video: {filename} ---")
                cap = cv2.VideoCapture(file_path)
                frame_count = 0
            
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                
                    if frame_count % 15 == 0: # Increased interval for faster display
                        analyze_frame(frame)
                        cv2_imshow(frame) # Display frames with boxes
                
                    frame_count += 1
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
            
                cap.release()

--- Execution ---
Update this with your actual folder path

    dataset_path = "/content/dataset" 
    process_dataset(dataset_path)

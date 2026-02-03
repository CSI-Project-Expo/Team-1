#  Face Detection

%pip install opencv-python deepface tf-keras
import cv2
import os
from deepface import DeepFace
from google.colab.patches import cv2_imshow

def process_dataset(directory_path):
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    video_exts = ('.mp4', '.avi', '.mov', '.mkv')

    # Iterate through every file in the folder
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        # Check if it's an image
        if filename.lower().endswith(image_exts):
            print(f"--- Processing Image: {filename} ---")
            frame = cv2.imread(file_path)
            if frame is not None:
                bit, emotion = analyze_face(frame)
                display_result(frame, f"Image: {filename}")

        # Check if it's a video
        elif filename.lower().endswith(video_exts):
            print(f"--- Processing Video: {filename} ---")
            cap = cv2.VideoCapture(file_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                bit, emotion = analyze_face(frame)
                cv2_imshow(frame) # Changed from cv2.imshow
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()

    cv2.destroyAllWindows()

def analyze_face(frame):
    """
    Core analysis logic (same as previous script)
    """
    try:
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        unsafe_emotions = {'angry', 'sad', 'fear', 'disgust'}
        safety_bit = 0

        for res in results:
            if res['dominant_emotion'] in unsafe_emotions:
                safety_bit = 1
                print(f"⚠️ ALERT: Safety Bit 1 for {res['dominant_emotion']}")
            else:
              print("The person is normal")
        return safety_bit, results[0]['dominant_emotion']
    except:
        return 0, "Error"

def display_result(frame, window_name):
    cv2_imshow(frame) # Changed from cv2.imshow
    cv2.waitKey(2000) # Show image for 2 seconds

RUN THE IMPORT

dataset_path = "import path"
process_dataset(dataset_path)

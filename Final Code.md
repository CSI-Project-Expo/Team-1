Install the required Libraries

    %pip install opencv-python deepface tf-keras
    %pip install opencv-python dlib numpy
    !pip install SpeechRecognition pydub moviepy
    !
    !apt-get update && apt-get install -y portaudio19-dev
    !pip install PyAudio
Import the Libraries

    import cv2
    import dlib
    import os
    import numpy as np
    from deepface import DeepFace
    from collections import Counter
    from pydub import AudioSegment
    import speech_recognition as sr
    from moviepy.editor import VideoFileClip
    import threading
    import queue
    import time
    from google.colab.patches import cv2_imshow
    from google.colab.output import eval_js, register_callback
    from base64 import b64decode, b64encode
    from IPython.display import display, Javascript
    import io
    import base64

Global flag to signal stopping

    stop_signal = False

Global queue for audio chunks and thread management

    audio_queue = queue.Queue()
    audio_processing_thread = None
    audio_stop_event = threading.Event()

    def stop_stream_callback():
        global stop_signal
        stop_signal = True
        print("Stopping stream via button...")
    
JavaScript to start the webcam and capture frames, with audio

    JS_CODE = """
    var video;
    var canvas;
    var div = null;
    var stream = null;
    var mediaRecorder = null;
    var audioChunks = [];

    async function startWebcam() {
      if (div) { div.remove(); } // Remove existing div if any
      video = document.createElement('video');
      video.style.display = 'block';
      video.width = 640;
      video.height = 480;

      try {
        stream = await navigator.mediaDevices.getUserMedia({video: true, audio: true});
      } catch (err) {
        console.error("Error accessing camera/microphone: ", err);
        alert("Could not access camera/microphone. Please ensure permissions are granted.");
        return;
      }

      div = document.createElement('div');
      div.appendChild(video);
    
      var button = document.createElement('button');
      button.textContent = 'Stop Stream';
      button.onclick = () => {
        google.colab.kernel.invokeFunction('stop_stream', []);
        if (stream) {
          stream.getTracks().forEach(track => track.stop());
          if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
          }
        }
        if (div) { div.remove(); } // Remove the video element from the DOM
      };
      div.appendChild(button);

      document.body.appendChild(div);
      video.srcObject = stream;
      await video.play();

      canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      startAudioRecording();
    }

    function startAudioRecording() {
      if (!stream) {
        console.error("Stream not available for audio recording.");
        return;
      }

      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];

      mediaRecorder.ondataavailable = async event => {
        audioChunks.push(event.data);
        if (audioChunks.length > 0) {
            const audioBlob = new Blob(audioChunks, { 'type' : 'audio/webm' });
            audioChunks = []; // Clear chunks after sending

            const reader = new FileReader();
            reader.readAsDataURL(audioBlob);
            reader.onloadend = () => {
                const base64data = reader.result;
                google.colab.kernel.invokeFunction('process_audio_chunk', [base64data]);
            }
        }
      };

      mediaRecorder.start(5000); // Emit data every 5 seconds
      console.log("Audio recording started, emitting data every 5 seconds.");
    }

    async function captureFrame() {
      if (!stream || !video.srcObject) {
        return ''; // Return empty if stream is not active
      }
      canvas.getContext('2d').drawImage(video, 0, 0);
      return canvas.toDataURL('image/jpeg', 0.8);
    }
    """
Converts the JavaScript frame to an OpenCV image

    def js_to_image(js_reply):
        image_bytes = base64.b64decode(js_reply.split(',')[1])
        jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
        return cv2.imdecode(jpg_as_np, flags=1)


--- INITIALIZATION & CONFIGURATION ---

    if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
        print("Downloading dlib model...")
        !wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        import time
        time.sleep(1) # Add a small delay
        !bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    HARSH_WORDS = ["leave", "help", "weired", "danger", "stop", "attack", "scared", "kill"]
    SHOUTING_THRESHOLD_DBFS = -15.0

--- CORE PROCESSING FUNCTIONS ---

 Calculates a movement score based on mouth opening distance

    def get_facial_movement_score(landmarks):
        points = landmarks.parts()
        mouth_top = points[51].y
        mouth_bottom = points[57].y
        return abs(mouth_bottom - mouth_top)
Analyzes a single frame for emotions and movements

    def analyze_visuals(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        unsafe_emotions = {'angry', 'sad', 'fear', 'disgust'}
        emotions_in_frame = []

        try:
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            for res in results:
                dom_emotion = res['dominant_emotion']
                emotion_confidence = res['emotion'][dom_emotion]
                emotions_in_frame.append(dom_emotion)

                region = res['region']
                x, y, w, h = region['x'], region['y'], region['w'], region['h']

                box_color = (0, 255, 0) # Green
                status_text = ""
                is_dangerous_movement = False

                try:
                    dlib_rect = dlib.rectangle(x, y, x + w, y + h)
                    landmarks = predictor(gray, dlib_rect)
                    movement_score = get_facial_movement_score(landmarks)
                    if movement_score > 79:
                        box_color = (0, 0, 255) # Red
                        status_text = "DANGEROUS MOVEMENT!"
                        is_dangerous_movement = True
                except: pass

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

Add conditional print statements

                if box_color == (0, 0, 255) or box_color == (0, 255, 255):
                    print(f"   Alert triggered: {status_text}")
                elif box_color == (0, 255, 0):
                    print("You are Safe! Keep going")

            if len(emotions_in_frame) > 1:
                counts = Counter(emotions_in_frame)
                for emo, count in counts.items():
                    if count > 1 and emo in unsafe_emotions:
                        print(f"🚨 GROUP ALERT: {count} people showing '{emo}'!")
        except: pass
        return frame
Analyzes a raw audio data blob for harsh words and shouting

    def analyze_audio_segment_data(audio_data_blob):
        recognizer = sr.Recognizer()
        is_shouting = False
        found_words = []
        text = ""

        try:
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_data_blob), format="webm") # Assuming webm from JS
        
            is_shouting = audio_segment.dBFS > SHOUTING_THRESHOLD_DBFS

            wav_data = io.BytesIO()
            audio_segment.export(wav_data, format="wav")
            wav_data.seek(0) # Reset stream position

            with sr.AudioFile(wav_data) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data).lower()
                found_words = [word for word in HARSH_WORDS if word in text]

            return text, found_words, is_shouting

        except sr.UnknownValueError:
            return None, [], False
        except Exception as e:
            print(f"Audio processing error: {e}")
            return None, [], False
Callback from JS to receive base64 audio chunks

    def audio_chunk_callback(base64_audio_data):
        global audio_queue
        try:
            header, encoded = base64_audio_data.split(",", 1)
            decoded_audio = base64.b64decode(encoded)
            audio_queue.put(decoded_audio)
        except Exception as e:
            print(f"Error decoding audio chunk: {e}")
Function to run in a separate thread for audio analysis

    def audio_processor_loop(stop_event):
        global audio_queue
        print("Audio processing thread started.")
        while not stop_event.is_set():
            try:
                audio_data_blob = audio_queue.get(timeout=1) # Get with timeout
                print(f"   [AUDIO ANALYSIS] Processing audio chunk ({len(audio_data_blob)} bytes)...") # Debug print
                text, words, is_shouting = analyze_audio_segment_data(audio_data_blob)

                audio_alerts = []
                if is_shouting and words:
                    audio_alerts.append("⚠️ SHOUTING DETECTED! Harsh words detected")
                elif is_shouting:
                    audio_alerts.append("⚠️ SHOUTING DETECTED! High volume.")
                elif words:
                    audio_alerts.append("⚠️ Harsh words detected")

                if audio_alerts:
                    print(f"\033[1m\033[91m   [AUDIO ANALYSIS] {' '.join(audio_alerts)} Transcription: {text if text else 'N/A'}\033[0m")
                elif text:
                    print(f"   [AUDIO ANALYSIS] [✅ CLEAN] Transcription: {text}")
                else:
                    print(f"   [AUDIO ANALYSIS] [NO SPEECH/HARSH WORDS DETECTED]") # Debug print

            except queue.Empty:
                continue # No audio in queue, check stop_event again
            except Exception as e:
                print(f"Error in audio processing loop: {e}")
        print("Audio processing thread stopped.")
Extracts text, checks for harsh words, and detects shouting in audio/video files.

    def process_audio_file(file_path):
        recognizer = sr.Recognizer()
        temp_wav = "temp_audio_conversion.wav"
        audio_segment = None
        is_shouting = False

        try:
            if file_path.lower().endswith(('.mp4', '.mkv', '.mov', '.avi')):
                video = VideoFileClip(file_path)
                if video.audio is None: return None, [], False
                video.audio.write_audiofile(temp_wav, codec='pcm_s16le', verbose=False, logger=None)
                audio_segment = AudioSegment.from_file(temp_wav)
            else:
                audio_segment = AudioSegment.from_file(file_path)
                audio_segment.export(temp_wav, format="wav")

            if audio_segment:
                is_shouting = audio_segment.dBFS > SHOUTING_THRESHOLD_DBFS

            found_words = []
            text = ""
            with sr.AudioFile(temp_wav) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data).lower()
                found_words = [word for word in HARSH_WORDS if word in text]
    
            return text, found_words, is_shouting

        except sr.UnknownValueError:
            return None, [], False
        except Exception:
            return None, [], False
        finally:
            if os.path.exists(temp_wav): os.remove(temp_wav)

--- MODE 1: DATASET PROCESSING ---

Processes images, videos, and audio files in a directory
    
    def run_dataset_analysis(folder_path):
        if not os.path.exists(folder_path):
            print("❌ Folder not found.")
            return

        image_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        video_exts = ('.mp4', '.avi', '.mov', '.mkv')
        audio_only_exts = ('.wav', '.mp3', '.m4a')

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            ext = os.path.splitext(filename)[1].lower()
            print(f"\nℹ️ ANALYZING: {filename}")

            if ext in image_exts:
                img = cv2.imread(file_path)
                if img is not None:
                    processed_img = analyze_visuals(img)
                    cv2_imshow(processed_img)

            elif ext in video_exts:
                text, words, is_shouting = process_audio_file(file_path)
                audio_alerts = []
                if is_shouting and words:
                    audio_alerts.append(f"⚠️ SHOUTING IN ANGER! Harsh words: {', '.join(words)}")
                elif is_shouting:
                    audio_alerts.append("⚠️ SHOUTING DETECTED! High volume.")
                elif words:
                    audio_alerts.append(f"⚠️ AUDIO ALERT! Harsh words found: {', '.join(words)}")

                if audio_alerts:
                    print(f"   {' '.join(audio_alerts)}")

                cap = cv2.VideoCapture(file_path)
                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    if frame_count % 30 == 0: # Increased interval to 30 for performance
                        processed_frame = analyze_visuals(frame)
                        cv2_imshow(processed_frame)
                    frame_count += 1
                cap.release()

            elif ext in audio_only_exts:
                text, words, is_shouting = process_audio_file(file_path)
                audio_alerts = []
                if is_shouting and words:
                    audio_alerts.append(f"⚠️ SHOUTING IN ANGER! Harsh words: {', '.join(words)}")
                elif is_shouting:
                    audio_alerts.append("⚠️ SHOUTING DETECTED! High volume.")
                elif words:
                    audio_alerts.append(f"⚠️ AUDIO ALERT! Harsh words found: {', '.join(words)}")

                if audio_alerts:
                    print(f"   [{' '.join(audio_alerts)}] Transcription: {text if text else 'N/A'}")
                elif text:
                    print(f"   [✅ CLEAN] Transcription: {text}")
                else:
                    print(f"   [❌ NO AUDIO/SPEECH DETECTED]")

        print("\n✅ Dataset Analysis Complete.")

--- MODE 2: LIVE STREAM PROCESSING ---

Accesses system camera and microphone for real-time analysis (Colab compatible)

    def start_live_stream():
        global stop_signal, audio_processing_thread, audio_queue, audio_stop_event
        print("Live detection started. (DeepFace + Dlib + Audio Analysis)")

        display(Javascript(JS_CODE))
        eval_js('startWebcam()') # This will now also start audio recording in JS
        register_callback('stop_stream', stop_stream_callback)
        register_callback('process_audio_chunk', audio_chunk_callback)

        audio_stop_event.clear() # Ensure event is clear at start
        audio_processing_thread = threading.Thread(target=audio_processor_loop, args=(audio_stop_event,))
        audio_processing_thread.daemon = True # Allow main program to exit even if thread is running
        audio_processing_thread.start()

        try:
            stop_signal = False # Ensure stop_signal is reset at the start of the try block
            while True:
                if stop_signal:
                    print("Exiting live stream loop.")
                    break

                js_frame = eval_js('captureFrame()')

                if not js_frame:
                    if stop_signal: # User clicked stop button and JS also stopped
                        break
                    else: # Unexpected error or stream closed from browser
                        print("Warning: No frame captured from webcam, stream might be disconnected. Stopping.")
                        break # Or handle error more robustly

                frame = js_to_image(js_frame)

                processed_frame = analyze_visuals(frame)
                cv2_imshow(processed_frame) # Display the frame with annotations

        except KeyboardInterrupt:
            print("Stream stopped by KeyboardInterrupt.")
        finally:
            stop_signal = False
            audio_stop_event.set() # Signal audio thread to stop
            if audio_processing_thread and audio_processing_thread.is_alive():
                audio_processing_thread.join(timeout=5) # Wait for thread to finish
                if audio_processing_thread.is_alive():
                    print("Warning: Audio thread did not terminate gracefully.")
            while not audio_queue.empty():
                try:
                    audio_queue.get_nowait()
                except queue.Empty:
                    break
            print("Live detection session ended.")

--- MAIN INTERFACE ---

    if __name__ == "__main__":
        print("--- Safety Analysis System ---")
        print("1. Analyze Dataset (Folder)")
        print("2. Start Live Stream (Camera/Mic)")

        choice = input("Select an option (1 or 2): ")

        if choice == '1':
            path = input("Enter the folder path: ")
            run_dataset_analysis(path)
        elif choice == '2':
            start_live_stream()
        else:
            print("Invalid choice.")

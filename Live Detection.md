
--- SETUP MODELS ---

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    stop_signal = False

    def stop_stream_callback():
        global stop_signal
        stop_signal = True
        print("Stopping stream via button...")

Register the Python function to be called from JavaScript
    
    register_callback('stop_stream', stop_stream_callback)

    JS_CODE = """
    var video;
    var canvas;
    var div = null;
    var stream = null;

    async function startWebcam() {
      if (div) { div.remove(); } // Remove existing div if any
      video = document.createElement('video');
      video.style.display = 'block';
      video.width = 640;
      video.height = 480;

      try {
        stream = await navigator.mediaDevices.getUserMedia({video: true});
      } catch (err) {
        console.error("Error accessing camera: ", err);
        alert("Could not access camera. Please ensure camera permissions are granted.");
        return;
      }

      div = document.createElement('div');
      div.appendChild(video);

Add a stop button

      var button = document.createElement('button');
      button.textContent = 'Stop Stream';
      button.onclick = () => {

        google.colab.kernel.invokeFunction('stop_stream', []);
        if (stream) {
          stream.getTracks().forEach(track => track.stop()); // Stop video stream in browser
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
        image_bytes = b64decode(js_reply.split(',')[1])
        jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
        return cv2.imdecode(jpg_as_np, flags=1)

Logic to detect unusual mouth movement
    
    def get_movement_score(landmarks):
        points = landmarks.parts()
        return abs(points[57].y - points[51].y)

Initialize Webcam

    display(Javascript(JS_CODE))
    eval_js('startWebcam()')

    print("Live detection started. (DeepFace + Dlib)")
    unsafe_emotions = {'angry', 'sad', 'fear', 'disgust'}

    try:
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
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

1. Emotion Detection (DeepFace)

            try:
                results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
                for res in results:
                    emotion = res['dominant_emotion']
                    x, y, w, h = res['region']['x'], res['region']['y'], res['region']['w'], res['region']['h']

   2. Movement Detection (Dlib)
      
                    rect = dlib.rectangle(x, y, x + w, y + h)
                    status_text = emotion.capitalize()
                    color = (0, 255, 0) # Normal

                    try:
                        landmarks = predictor(gray, rect)
                        if get_movement_score(landmarks) > 60:
                            status_text = "UNUSUAL MOVEMENT"
                            color = (0, 0, 255)
                    except:
                        pass

                    if emotion in unsafe_emotions and status_text != "UNUSUAL MOVEMENT":
                        color = (0, 0, 255)
                        status_text = f"ALERT: {emotion.upper()}"

                    elif emotion == 'neutral':
                            color = (0, 255, 255)

                    if color == (0, 0, 255):
                        print(f"Alert triggered: {status_text}")

                    elif color == (0, 255, 255):
                            print("Neutral Emotion Alert")

                    else:
                        print(f"Status: {status_text}")

            except Exception as e:
                continue
      
To stop the live stream

    except KeyboardInterrupt:
        print("Stream stopped by KeyboardInterrupt.")
   
    finally:
        stop_signal = False
        print("Live detection session ended.")

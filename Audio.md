Install SpeechRecognition library

    !pip install SpeechRecognition pydub moviepy
Additional Libraries

    import speech_recognition as sr
    from moviepy.editor import VideoFileClip
    from pydub import AudioSegment

    HARSH_WORDS = ["leave", "help", "weired", "danger", "stop", "attack","scared","kill"]
  Handles both Video and Audio files to extract text
  
    def process_file(file_path):
        recognizer = sr.Recognizer()
        temp_wav = "temp_audio_conversion.wav"

        try:

            if file_path.lower().endswith(('.mp4', '.mkv', '.mov', '.avi')):
                print(f"🎥 Extracting audio from video: {os.path.basename(file_path)}")
                video = VideoFileClip(file_path)
                if video.audio is None:
                    return None, "NO_AUDIO_TRACK_FOUND" 
                video.audio.write_audiofile(temp_wav, codec='pcm_s16le', verbose=False, logger=None)
                audio_target = temp_wav
            else:
                audio = AudioSegment.from_file(file_path)
                audio.export(temp_wav, format="wav")
                audio_target = temp_wav

            with sr.AudioFile(audio_target) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data).lower()

                found = [word for word in HARSH_WORDS if word in text]
                return text, found

        except sr.UnknownValueError:
            return None, "NO_SPEECH_RECOGNIZED"
        except sr.RequestError as e:
            return None, f"API_REQUEST_ERROR: {e}"
        except Exception as e:
            return None, f"GENERIC_PROCESSING_ERROR: {str(e)}"
        finally:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
Iterates through a folder and processes all media files

    def analyze_dataset(folder_path):
        if not os.path.exists(folder_path):
            print("❌ Folder not found. Please check the path.")
            return

        print(f"🚀 Starting analysis on dataset: {folder_path}\n" + "="*40)

        results = []
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            if filename.lower().endswith(('.mp4', '.mkv', '.mov', '.avi', '.wav', '.mp3', '.m4a')):
                print(f"🔍 Analyzing: {filename}...")
                transcription, detection_status = process_file(file_path)

                if transcription:
                    status_text = f"⚠️ ALERT! DETECTED: {', '.join(detection_status)}" if detection_status else "✅ CLEAN"
                    print(f"   [{status_text}]")
                    results.append({"file": filename, "text": transcription, "harsh": detection_status})
                else:
                    if detection_status == "NO_AUDIO_TRACK_FOUND" or detection_status == "NO_SPEECH_RECOGNIZED":
                        print(f"   ❌ No audio detected in {filename}")
                    elif detection_status.startswith("API_REQUEST_ERROR"):
                        print(f"   ❌ API Request Error for {filename}: {detection_status.replace('API_REQUEST_ERROR: ', '')}")
                    elif detection_status.startswith("GENERIC_PROCESSING_ERROR"):
                        print(f"   ❌ Failed to process {filename}: {detection_status.replace('GENERIC_PROCESSING_ERROR: ', '')}")
                    else:
                        print(f"   ❌ An unknown error occurred while processing {filename}: {detection_status}")

        print("\n" + "="*40 + "\n✅ Dataset Analysis Complete.")
        return results
Run the function below:

    dataset_results = analyze_dataset('/content/dataset')

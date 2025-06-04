import cv2
import time
import os
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv
import datetime
import sounddevice as sd
import numpy as np

# Load environment variables
load_dotenv()

# Cloudinary config
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_NAME"),
    api_key=os.getenv("CLOUDINARY_KEY"),
    api_secret=os.getenv("CLOUDINARY_SECRET")
)

# Sound detection parameters
SAMPLE_RATE = 44100  # Sampling rate
DURATION = 0.5       # Duration to record per check (seconds)
THRESHOLD = 0.1      # Volume threshold (adjust as needed)

def detect_close_sound():
    print("Listening for sound...")

    while True:
        audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()

        volume_norm = np.linalg.norm(audio_data)
        print(f"Detected volume: {volume_norm:.3f}")
        print(volume_norm)
        
        if volume_norm > THRESHOLD:
            print("🔊 Loud sound detected!")
            return True

# Initialize webcam
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Webcam not accessible.")
    exit()

print("System ready. Waiting for sound...")

try:
    while True:
        if detect_close_sound():
            print("📸 Starting 1-minute capture session...")
            start_time = time.time()
            
            while time.time() - start_time < 60:
                ret, frame = cam.read()
                if not ret:
                    print("Failed to capture image.")
                    continue

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename ="image.jpg"
                cv2.imwrite(filename, frame)

                try:
                    response = cloudinary.uploader.upload(
                        filename,
                        folder="hajoj/",
                        use_filename=True,
                        overwrite=True,
                        resource_type="image"
                    )
                    print(f"✅ Uploaded: {response['secure_url']}")
                except Exception as e:
                    print(f"❌ Upload failed: {e}")

                time.sleep(0.5)

            print("🛑 1-minute session ended. Listening for next sound...")

except KeyboardInterrupt:
    print("👋 Stopping...")

finally:
    cam.release()
    cv2.destroyAllWindows()

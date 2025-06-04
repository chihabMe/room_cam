import cv2
import time
import os
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv
import datetime
import pyaudio
import numpy as np

# Load environment variables
load_dotenv()

# Cloudinary config
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_NAME"),
    api_key=os.getenv("CLOUDINARY_KEY"),
    api_secret=os.getenv("CLOUDINARY_SECRET")
)

# Audio config
CHUNK = 1024  # Number of audio samples per frame
RATE = 44100  # Sampling rate
THRESHOLD = 500  # Sensitivity threshold (adjust if needed)

def detect_close_sound():
    print("Listening for sound...")
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    try:
        while True:
            data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
            volume = np.linalg.norm(data)
            if volume > THRESHOLD:
                print(f"Sound detected! Volume: {volume:.2f}")
                return True
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

# Initialize webcam
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Webcam not accessible.")
    exit()

print("System ready. Waiting for sound...")

try:
    while True:
        if detect_close_sound():
            print("Starting 1-minute capture session...")
            start_time = time.time()
            
            while time.time() - start_time < 60:
                ret, frame = cam.read()
                if not ret:
                    print("Failed to capture image.")
                    continue

                # Save with timestamp
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"frame_{timestamp}.jpg"
                cv2.imwrite(filename, frame)

                # Upload to Cloudinary
                try:
                    response = cloudinary.uploader.upload(
                        filename,
                        folder="hajoj/",
                        use_filename=True,
                        overwrite=False,
                        resource_type="image"
                    )
                    print(f"Uploaded to Cloudinary: {response['secure_url']}")
                except Exception as e:
                    print(f"Upload failed: {e}")

                time.sleep(0.5)

            print("1-minute session ended. Listening for next sound...")

except KeyboardInterrupt:
    print("Stopping...")

finally:
    cam.release()
    cv2.destroyAllWindows()

import cv2
import time
import os
import cloudinary
import cloudinary.uploader
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Cloudinary configuration
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_NAME"),
    api_key=os.getenv("CLOUDINARY_KEY"),
    api_secret=os.getenv("CLOUDINARY_SECRET")
)

# Initialize webcam
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Webcam not accessible.")
    exit()

# Sound detection parameters
SOUND_THRESHOLD = 0.3  # Adjust this if too sensitive or not sensitive enough
LISTEN_DURATION = 0.5  # Seconds of audio to analyze per chunk
sample_rate = 44100

def detect_sound(threshold=SOUND_THRESHOLD):
    audio = sd.rec(int(LISTEN_DURATION * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    volume = np.linalg.norm(audio)  # Calculate root mean square
    return volume > threshold

def take_and_upload_photo():
    ret, frame = cam.read()
    if not ret:
        print("Failed to capture image.")
        return

    filename = "frame.jpg"
    cv2.imwrite(filename, frame)

    try:
        response = cloudinary.uploader.upload(
            filename,
            folder="hajoj/",
            use_filename=True,
            overwrite=True,
            resource_type="image"
        )
        print(f"Uploaded to Cloudinary: {response['secure_url']}")
    except Exception as e:
        print(f"Upload failed: {e}")

print("Listening for loud sounds...")

try:
    while True:
        if detect_sound():
            print("🔊 Loud sound detected! Capturing photo...")
            take_and_upload_photo()
            time.sleep(2)  # Prevent spamming multiple captures instantly

except KeyboardInterrupt:
    print("Stopping...")

finally:
    cam.release()
    cv2.destroyAllWindows()

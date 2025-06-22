# import cv2
# import numpy as np
# import sounddevice as sd
# from scipy.io.wavfile import write
# import datetime
# import threading
# import time
# import webbrowser
# import uuid
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
# import smtplib

# class HRInterviewSystem:
#     def __init__(self):
#         # Initialize face detection
#         self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#         self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
#         # Audio recording parameters
#         self.sample_rate = 44100
#         self.is_recording = False
#         self.audio_thread = None
        
#         # Initialize webcam
#         self.cap = cv2.VideoCapture(0)
        
#         # Notification states
#         self.fake_detected_count = 0
#         self.real_count = 0
#         self.notification_threshold = 10
        
#         # Meeting details
#         self.meeting_id = None
#         self.meeting_url = None
        
#         # Previous face position for movement analysis
#         self.prev_face_pos = None
#         self.emotion_history = []
        
#         # Email configuration (replace with your details)
#         self.smtp_server = 'smtp.gmail.com'
#         self.smtp_port = 587
#         self.email = 'your_email@gmail.com'
#         self.email_password = 'your_app_password'

#     def create_meeting(self, interviewer_email, candidate_email):
#         """Create a Google Meet meeting and send invitations"""
#         self.meeting_id = str(uuid.uuid4())[:8]
#         self.meeting_url = f"https://meet.google.com/{self.meeting_id}"
        
#         # Send meeting invitations
#         subject = "HR Interview Meeting Invitation"
#         body = f"""
#         You are invited to an HR interview meeting.
        
#         Meeting URL: {self.meeting_url}
#         Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
        
#         Please join using the link above.
#         """
        
#         # Send to both HR and candidate
#         self.send_email(interviewer_email, subject, body)
#         self.send_email(candidate_email, subject, body)
        
#         return self.meeting_url

#     def send_email(self, recipient, subject, body):
#         """Send email notification"""
#         msg = MIMEMultipart()
#         msg['From'] = self.email
#         msg['To'] = recipient
#         msg['Subject'] = subject
#         msg.attach(MIMEText(body, 'plain'))
        
#         try:
#             with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
#                 server.starttls()
#                 server.login(self.email, self.email_password)
#                 server.send_message(msg)
#         except Exception as e:
#             print(f"Failed to send email: {str(e)}")

#     def record_audio(self):
#         """Record audio during the interview"""
#         while self.is_recording:
#             # Record 2 seconds of audio at a time
#             recording = sd.rec(int(2 * self.sample_rate), 
#                             samplerate=self.sample_rate, 
#                             channels=1)
#             sd.wait()
            
#             # Save audio with timestamp
#             timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#             filename = f"interview_audio_{timestamp}.wav"
#             write(filename, self.sample_rate, recording)
#             time.sleep(1)

#     def analyze_facial_features(self, frame):
#         """Analyze facial features for deepfake detection"""
#         # Convert to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
#         # Detect faces
#         faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
#         if len(faces) > 0:
#             (x, y, w, h) = faces[0]
#             face_roi = frame[y:y+h, x:x+w]
#             gray_roi = gray[y:y+h, x:x+w]
            
#             # Detect eyes
#             eyes = self.eye_cascade.detectMultiScale(gray_roi)
            
#             # Analyze natural movement
#             if self.prev_face_pos:
#                 prev_x, prev_y = self.prev_face_pos
#                 movement = abs(x - prev_x) + abs(y - prev_y)
#                 if 2 < movement < 30:  # Natural movement range
#                     self.real_count += 1
#                     self.fake_detected_count = max(0, self.fake_detected_count - 1)
#                 else:
#                     self.fake_detected_count += 1
#                     self.real_count = max(0, self.real_count - 1)
            
#             self.prev_face_pos = (x, y)
            
#             # Calculate lip movement using color changes in mouth region
#             mouth_roi = face_roi[int(h*0.6):int(h*0.8), int(w*0.3):int(w*0.7)]
#             avg_color = np.mean(mouth_roi)
            
#             if hasattr(self, 'prev_mouth_color'):
#                 lip_movement = abs(avg_color - self.prev_mouth_color)
#                 if lip_movement > 5:  # Threshold for lip movement
#                     self.real_count += 1
            
#             self.prev_mouth_color = avg_color
            
#             return True, (x, y, w, h), eyes
#         return False, None, None

#     def send_notification(self, is_fake, email):
#         """Send notification about authenticity"""
#         subject = "Interview Authenticity Alert"
#         if is_fake:
#             body = "⚠️ WARNING: Potential deepfake detected in the ongoing interview!"
#         else:
#             body = "✅ Confirmation: Real person verified in the interview."
        
#         self.send_email(email, subject, body)

#     def start_interview(self, hr_email, candidate_email=None):
#         """Start the interview monitoring"""
#         # Create and send meeting link if emails are provided
#         if hr_email and candidate_email:
#             meeting_url = self.create_meeting(hr_email, candidate_email)
#             print(f"\nMeeting URL: {meeting_url}")
#             webbrowser.open(meeting_url)
        
#         # Start audio recording
#         self.is_recording = True
#         self.audio_thread = threading.Thread(target=self.record_audio)
#         self.audio_thread.start()
        
#         try:
#             while True:
#                 ret, frame = self.cap.read()
#                 if not ret:
#                     break
                
#                 # Analyze facial features
#                 has_face, face_rect, eyes = self.analyze_facial_features(frame)
                
#                 if has_face:
#                     (x, y, w, h) = face_rect
#                     # Draw rectangle around face
#                     cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
#                     # Draw eyes
#                     if eyes is not None:
#                         for (ex, ey, ew, eh) in eyes:
#                             cv2.rectangle(frame[y:y+h, x:x+w], (ex, ey), 
#                                         (ex+ew, ey+eh), (0, 255, 0), 2)
                    
#                     # Display lip movement status
#                     if hasattr(self, 'prev_mouth_color'):
#                         cv2.putText(frame, "Lip Movement Detected", 
#                                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
#                                   0.7, (255, 255, 0), 2)
                    
#                     # Show authenticity status
#                     if self.fake_detected_count >= self.notification_threshold:
#                         cv2.putText(frame, "FAKE DETECTED!", (10, 30),
#                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#                         self.send_notification(True, hr_email)
#                         if candidate_email:
#                             self.send_notification(True, candidate_email)
                    
#                     elif self.real_count >= self.notification_threshold:
#                         cv2.putText(frame, "REAL PERSON", (10, 30),
#                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#                         self.send_notification(False, hr_email)
                    
#                     # Show head movement status
#                     if self.prev_face_pos:
#                         cv2.putText(frame, "Head Movement Detected", (10, 120),
#                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
#                 else:
#                     cv2.putText(frame, "No face detected", (10, 30),
#                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
#                 cv2.imshow('HR Interview System', frame)
                
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break
        
#         finally:
#             self.cleanup()

#     def cleanup(self):
#         """Clean up resources"""
#         self.is_recording = False
#         if self.audio_thread:
#             self.audio_thread.join()
#         self.cap.release()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     # Example usage
#     hr_system = HRInterviewSystem()
#     hr_email = "sk93629611@gmail.com"  # Replace with actual HR email
#     candidate_email = "mohanraj4847@gmail.com"  # Replace with candidate email
#     hr_system.start_interview(hr_email, candidate_email)

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image
import numpy as np 
import os
import sys
from utils.face_detector import extract_face
import torch.nn as nn
import time

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load base model and modify the classifier
weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights)
model.classifier[1] = nn.Linear(model.last_channel, 2)  # 2 classes: Real / Fake

# Load trained weights
model.load_state_dict(torch.load("E:\Deepfake Detection\model\mobilenet_deepfake.pth", map_location=device))
model = model.to(device)
model.eval()

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
])

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

with torch.no_grad():
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        try:
            face = extract_face(frame)

            if isinstance(face, np.ndarray):
                img = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img)
                input_tensor = preprocess(img_pil).unsqueeze(0).to(device)

                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0][predicted_class].item()

                label = "Fake" if predicted_class == 1 else "Real"
                color = (0, 0, 255) if label == "Fake" else (0, 255, 0)
                confidence_text = f"{label} ({confidence * 100:.2f}%)"
            else:
                raise ValueError("No valid face detected")

        except Exception as e:
            print("Face detection error:", e)
            confidence_text = "No face detected"
            color = (255, 255, 0)

        # Display result
        cv2.putText(frame, confidence_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, color, 2)

        # Optional: show FPS
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (100, 255, 100), 1)

        cv2.imshow("Deepfake Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

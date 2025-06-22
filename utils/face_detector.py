import cv2
import dlib
import numpy as np
from imutils import face_utils

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("E:\Deepfake Detection\model\shape_predictor_68_face_landmarks.dat")

def extract_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    
    if len(rects) == 0:
        return None
    
    # Get the largest face
    rect = max(rects, key=lambda r: (r.right()-r.left())*(r.bottom()-r.top()))
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    
    # Expand the face region slightly
    y = max(0, y - int(h * 0.15))
    x = max(0, x - int(w * 0.15))
    h = min(frame.shape[0] - y, int(h * 1.3))
    w = min(frame.shape[1] - x, int(w * 1.3))
    
    face = frame[y:y+h, x:x+w]
    return face
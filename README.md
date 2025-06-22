# Deepfake Detection Demo

This project is a demo application for detecting deepfakes in live video calls using a Streamlit web interface. It includes placeholder detection logic and a training script for a MobileNetV2-based deepfake classifier.

## Features

- Streamlit UI for hosting and joining video calls (demo mode)
- Live webcam feed with simulated deepfake detection (head pose, lip movement, eye blinking, anomaly detection)
- Training script for MobileNetV2 deepfake classifier using images in `dataset/`
- Example dataset structure for real and fake images
- Model saving/loading support

## Project Structure

```
deepfake.py                # Streamlit app for live detection demo
detect.py                  # (Presumed) Script for running detection on images/videos
train_mobilenetv2.py       # Script to train MobileNetV2 classifier
dataset/
    fake/                  # Folder with fake/deepfake images
    real/                  # Folder with real images
model/
    deepfake_mobilenetv2.h5 # Trained Keras model (after training)
```

## Getting Started

### 1. Install dependencies

```sh
pip install -r requirements.txt
```

### 2. Run the Streamlit demo

```sh
streamlit run deepfake.py
```

### 3. Train the MobileNetV2 model

Make sure your dataset is organized as shown above, then run:

```sh
python train_mobilenetv2.py
```

The trained model will be saved to `model/deepfake_mobilenetv2.h5`.

## Notes

- The Streamlit demo uses placeholder detection logic. For real detection, integrate trained models and proper video processing.
- For actual peer-to-peer video calls, a WebRTC setup is required (not included in this demo).
- The dataset folders should contain images of faces for training and testing.

## Requirements

See [requirements.txt](requirements.txt).

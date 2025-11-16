import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os

# --------------------
# Load YOLO Tiny-ADAS
# --------------------
MODEL_PATH = "runs/tiny_adas/augmented_model/weights/best.pt"
model = YOLO(MODEL_PATH)

# --------------------
# Simple Fog Effect
# --------------------
def add_fog(img):
    h, w = img.shape[:2]
    fog = np.random.normal(loc=200, scale=30, size=(h, w, 3)).astype(np.uint8)
    foggy = cv2.addWeighted(img, 0.7, fog, 0.3, 0)
    return foggy

# --------------------
# Simple Rain Effect
# --------------------
def add_rain(img):
    rain = img.copy()
    for _ in range(300):
        x1 = np.random.randint(0, img.shape[1])
        y1 = np.random.randint(0, img.shape[0])
        length = np.random.randint(5, 15)
        cv2.line(rain, (x1, y1), (x1, y1 + length), (200, 200, 200), 1)
    rain = cv2.blur(rain, (3, 3))
    return rain

# --------------------
# YOLO Inference for Images
# --------------------
def detect_image(img):
    results = model.predict(img, conf=0.4)
    return results[0].plot()

# --------------------
# YOLO Inference for Videos
# --------------------
def detect_video(video_path):
    cap = cv2.VideoCapture(video_path)

    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out_path = temp_out.name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(3))
    height = int(cap.get(4))

    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, conf=0.4)
        out_frame = results[0].plot()
        out.write(out_frame)

    cap.release()
    out.release()
    return out_path

# --------------------
# STREAMLIT UI
# --------------------
st.title("Tiny-ADAS — Robustness Test Demo")
st.write("Upload an Image or Video → Apply Augmentation → Run Tiny ADAS YOLO Model")

option = st.radio("Choose Input Type:", ["Image", "Video"])

# --------------------
# IMAGE PIPELINE
# --------------------
if option == "Image":
    uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded:
        img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
        st.image(img, channels="BGR", caption="Original Image")

        aug = st.selectbox("Apply Augmentation:", ["None", "Fog", "Rain"])

        if aug == "Fog":
            img_aug = add_fog(img)
            st.image(img_aug, channels="BGR", caption="Fog-Augmented Image")
        elif aug == "Rain":
            img_aug = add_rain(img)
            st.image(img_aug, channels="BGR", caption="Rain-Augmented Image")
        else:
            img_aug = img

        if st.button("Run Detection"):
            detected = detect_image(img_aug)
            st.image(detected, channels="BGR", caption="Detection Result")

# --------------------
# VIDEO PIPELINE
# --------------------
if option == "Video":
    uploaded = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
    if uploaded:
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(uploaded.read())
        st.video(temp_video.name)

        aug = st.selectbox("Apply Augmentation:", ["None", "Fog", "Rain"])

        if aug == "Fog" or aug == "Rain":
            st.warning("⚠ Video augmentation is heavy — this demo only applies augmentation to first frame preview.")

            cap = cv2.VideoCapture(temp_video.name)
            ret, frame = cap.read()
            cap.release()

            if aug == "Fog":
                frame_aug = add_fog(frame)
            else:
                frame_aug = add_rain(frame)

            st.image(frame_aug[:, :, ::-1], caption="Preview of Augmented Frame")

        if st.button("Run Video Detection"):
            st.info("Processing video… please wait.")
            out_path = detect_video(temp_video.name)
            st.video(out_path)
            st.success("Done!")


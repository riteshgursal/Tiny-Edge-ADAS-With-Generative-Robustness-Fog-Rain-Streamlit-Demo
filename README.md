<h1>🚗 Tiny-Edge ADAS With Generative Robustness (Fog/Rain) — Streamlit Demo</h1>

A lightweight, edge-friendly ADAS system with generative weather augmentation and YOLO-based object detection.

---

**📝 Overview**

This project demonstrates a Tiny ADAS (Advanced Driver Assistance System) designed for edge computing, combined with generative robustness augmentation.
The system allows users to:

Upload images or videos

Apply fog or rain augmentation to simulate difficult driving environments

Run real-time object detection using a YOLOv8-nano model

Compare clear vs augmented detection performance

This creates a compact yet powerful pipeline suitable for small devices, embedded systems, and smart-mobility research.

---

**✨ Features**

**🔹 1. Weather Augmentation**

Add environmental challenges automatically:

Fog overlay

Rain streak simulation

These augmentations help evaluate how robust a model is under adverse driving conditions.

**🔹 2. Tiny-Edge ADAS Detection**

YOLOv8-nano is used for fast, lightweight detection:

Cars

People

Bikes

Road objects

Traffic-related entities

Works smoothly on CPU, making it edge-device-friendly.

**🔹 3. Image + Video Support**

The demo supports:

Images: real-time augmented visualization + detection

Videos: frame-by-frame processing with an output video showing ADAS predictions

---

**🗂️ Project Structure**

```
.
├── augment/
│   ├── fog_augment.py
│   ├── rain_augment.py
│   └── sample_images/          # add images here for testing
├── detector/
│   ├── train_detector.py       # YOLO training (optional)
│   ├── detect.py               # YOLO inference logic
│   └── data.yaml               # dataset config
├── streamlit_app/
│   └── app.py                  # Streamlit Demo
└── README.md

```
---

**🚀 How to Run the Streamlit Demo**

*1. Install Requirements*
```
pip install -r requirements.txt
```

*2. Run the Demo*
```
streamlit run streamlit_app/app.py
```

*3. Open Your Browser*

It automatically opens at:
```
http://localhost:8501
```

---

**🎮 How the Demo Works**

Select: Image / Video

Upload your media

Choose augmentation:

None

Fog

Rain

Preview appears instantly

Click Run Detection

See:

Bounding boxes

Augmented vs clear performance

Processed video output

This allows edge-ready ADAS evaluation in a compact tool.

---

**🔧 Tech Stack**

Python

YOLOv8-nano (Ultralytics)

OpenCV

NumPy & Pillow

Streamlit

Generative Weather Augmentation (custom code)

---

**📌 Use-Case Alignment (NCCU Project)**

This project demonstrates:

✔ Lightweight ADAS suitable for embedded / low-power devices
✔ Generative robustness using fog and rain
✔ Strong cross-modal understanding (vision + augmentation + detection)
✔ Real-time inference pipeline
✔ Deployable demo interface

Perfectly aligned with:

**Generative Robustness & Tiny-Edge ADAS for Smart Systems**

🧪 Example Outputs

Fog-augmented street images with detections

Rain-augmented highway frames

YOLO detection results on video sequences


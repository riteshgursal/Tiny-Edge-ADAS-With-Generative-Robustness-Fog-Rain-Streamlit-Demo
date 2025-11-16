from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO("runs/tiny_adas/augmented_model/weights/best.pt")

img_path = "test.jpg"
results = model(img_path)

results[0].show()

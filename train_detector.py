from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="detector/data.yaml",
    epochs=20,
    imgsz=640,
    project="runs/tiny_adas",
    name="augmented_model"
)

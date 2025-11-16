import pandas as pd
from ultralytics import YOLO

baseline = YOLO("runs/tiny_adas/baseline/weights/best.pt")
augmented = YOLO("runs/tiny_adas/augmented_model/weights/best.pt")

baseline_res = baseline.val()
aug_res = augmented.val()

data = {
    "Model": ["Baseline", "Augmented"],
    "mAP50": [baseline_res.box.map50, aug_res.box.map50],
    "mAP50-95": [baseline_res.box.map, aug_res.box.map]
}

df = pd.DataFrame(data)
df.to_csv("results/eval_table.csv", index=False)

print(df)

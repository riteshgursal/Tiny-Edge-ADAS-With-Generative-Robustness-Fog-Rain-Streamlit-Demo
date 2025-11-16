import cv2
import numpy as np
import os

def add_rain(img, drops=300, intensity=0.6):
    rain = np.zeros_like(img)
    h, w = img.shape[:2]

    for _ in range(drops):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        length = np.random.randint(10, 20)
        cv2.line(rain, (x, y), (x, y+length), (255, 255, 255), 1)

    rain = cv2.GaussianBlur(rain, (7, 7), 0)
    rain_img = cv2.addWeighted(img, 1, rain, intensity, 0)

    return rain_img

def augment_folder(src, dst):
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        if not f.endswith((".jpg", ".png")):
            continue
        img = cv2.imread(os.path.join(src, f))
        rainy = add_rain(img)
        cv2.imwrite(os.path.join(dst, f), rainy)

if __name__ == "__main__":
    augment_folder("augment/sample_images", "augment/rain_output")

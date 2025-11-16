import cv2
import os

def add_fog(img, fog_intensity=0.6):
    h, w = img.shape[:2]
    fog = cv2.resize(cv2.GaussianBlur(img, (51, 51), 0), (w, h))
    fog = cv2.addWeighted(img, 1 - fog_intensity, fog, fog_intensity, 0)
    return fog

def augment_folder(src, dst):
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        if not f.endswith((".jpg", ".png")):
            continue
        img = cv2.imread(os.path.join(src, f))
        foggy = add_fog(img)
        cv2.imwrite(os.path.join(dst, f), foggy)

if __name__ == "__main__":
    augment_folder("augment/sample_images", "augment/fog_output")

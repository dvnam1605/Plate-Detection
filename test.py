import glob
import random
import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO


all_images = glob.glob("data/archive/images/train/*.jpg")


test_images = random.sample(all_images, 6)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for img_path, ax in zip(test_images, axes.flatten()):
    results = model(img_path, device=0)      #  YOLO 
    result_img = results[0].plot() # annotated 
    
    img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)
    ax.axis("off")
    ax.set_title(os.path.basename(img_path))  

plt.tight_layout()
plt.show()

TRAINED_MODEL_PATH = 'runs/detect/train3/weights/best.pt' 

# Kiểm tra xem file có tồn tại không
if not os.path.exists(TRAINED_MODEL_PATH):
    raise FileNotFoundError(f"Không tìm thấy mô hình đã huấn luyện tại: {TRAINED_MODEL_PATH}")

model = YOLO(TRAINED_MODEL_PATH)


all_images = glob.glob("/kaggle/working/final_multiclass_dataset/images/val/*.jpg")
test_images = random.sample(all_images, 6)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for img_path, ax in zip(test_images, axes.flatten()):
    results = model(img_path, device='0')      
    result_img = results[0].plot()
    
    img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)
    ax.axis("off")
    ax.set_title(os.path.basename(img_path))  

plt.tight_layout()
plt.show()
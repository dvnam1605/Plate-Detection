import os
import shutil
from ultralytics import YOLO

model = YOLO("yolo11s.pt")

results = model.train(
    data = 'data/final_multiclass_dataset/data.yaml',
    epochs = 100,
    imgsz = 640,
    lr0 = 5e-4,
    augment = True,
    # device= [0,1],
    batch = 32,
    lrf = 0.1  
)

model.save("my_trainer.pt")


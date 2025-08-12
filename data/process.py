import os
import shutil
from ultralytics import YOLO
import yaml
from tqdm import tqdm

model = YOLO("yolo11s.pt")

SOURCE_DATA_DIR = 'data/archive'
NEW_DATA_DIR = 'data/final_multiclass_dataset' 
CONF_THRESHOLD = 0.25 

print("Đang tải mô hình COCO (yolov11n.pt)...")
coco_model = YOLO('yolo11n.pt')
print("Tải mô hình thành công.")


ALL_CLASSES = ['license_plate', 'person', 'car', 'motorcycle', 'bus']

# Định nghĩa các lớp COCO 
coco_names = coco_model.names
COCO_CLASSES_TO_MAP = {
    coco_names.get(0): ALL_CLASSES.index('person'),      
    coco_names.get(2): ALL_CLASSES.index('car'),         
    coco_names.get(3): ALL_CLASSES.index('motorcycle'),  
    coco_names.get(5): ALL_CLASSES.index('bus'),         
}
COCO_IDS_TO_DETECT = [k for k, v in coco_names.items() if v in COCO_CLASSES_TO_MAP]


def process_and_verify_labels(source_img_dir, source_label_dir, dest_img_dir, dest_label_dir, split_name):
    os.makedirs(dest_img_dir, exist_ok=True)
    os.makedirs(dest_label_dir, exist_ok=True)
    
    image_files = sorted(os.listdir(source_img_dir)) # Sắp xếp để kết quả log nhất quán
    
    for img_file in tqdm(image_files, desc=f"Đang xử lý tập {split_name}"):
        source_img_path = os.path.join(source_img_dir, img_file)
        source_label_path = os.path.join(source_label_dir, os.path.splitext(img_file)[0] + '.txt')

        shutil.copy(source_img_path, dest_img_dir)

        final_labels = []

        if os.path.exists(source_label_path):
            with open(source_label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts and int(parts[0]) == 0:
                        # Giữ nguyên nhãn license_plate (ID=0)
                        final_labels.append(line.strip())

        results = coco_model.predict(source_img_path, conf=CONF_THRESHOLD, device='0', verbose=False)
        
        for box in results[0].boxes:
            coco_class_id = int(box.cls[0])
            if coco_class_id in COCO_IDS_TO_DETECT:
                coco_class_name = coco_names[coco_class_id]
                new_class_id = COCO_CLASSES_TO_MAP[coco_class_name]
                x_center, y_center, width, height = box.xywhn[0]
                final_labels.append(f"{new_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        dest_label_path = os.path.join(dest_label_dir, os.path.splitext(img_file)[0] + '.txt')
        with open(dest_label_path, 'w') as f:
            for label in final_labels:
                f.write(label + '\n')
                
    # check 
    print(f"\n--- Kiểm tra 5 file nhãn đầu tiên của tập {split_name} ---")
    label_files_to_check = sorted(os.listdir(dest_label_dir))[:5]
    for label_file in label_files_to_check:
        print(f"\nNội dung file: {label_file}")
        with open(os.path.join(dest_label_dir, label_file), 'r') as f:
            print(f.read().strip())
    print("--------------------------------------------------\n")


# Xử lý tập train
process_and_verify_labels(
    source_img_dir=os.path.join(SOURCE_DATA_DIR, 'images/train'),
    source_label_dir=os.path.join(SOURCE_DATA_DIR, 'labels/train'),
    dest_img_dir=os.path.join(NEW_DATA_DIR, 'images/train'),
    dest_label_dir=os.path.join(NEW_DATA_DIR, 'labels/train'),
    split_name='train'
)

# Xử lý tập validation
process_and_verify_labels(
    source_img_dir=os.path.join(SOURCE_DATA_DIR, 'images/val'),
    source_label_dir=os.path.join(SOURCE_DATA_DIR, 'labels/val'),
    dest_img_dir=os.path.join(NEW_DATA_DIR, 'images/val'),
    dest_label_dir=os.path.join(NEW_DATA_DIR, 'labels/val'),
    split_name='val'
)

yaml_content = {
    'train': os.path.join('../', os.path.basename(NEW_DATA_DIR), 'images/train'),
    'val': os.path.join('../', os.path.basename(NEW_DATA_DIR), 'images/val'),
    'nc': len(ALL_CLASSES),
    'names': ALL_CLASSES
}
yaml_path = os.path.join(NEW_DATA_DIR, 'data.yaml')
with open(yaml_path, 'w') as f:
    yaml.dump(yaml_content, f, sort_keys=False, default_flow_style=None)

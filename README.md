# License Plate Detection with YOLO

A computer vision project for detecting license plates using YOLOv11 (YOLO version 11) deep learning model.

## 📋 Overview

This project implements an automated license plate detection system using the state-of-the-art YOLOv11 object detection model. The system can detect license plates in images and videos with high accuracy and real-time performance.

## 🔧 Features

- **License Plate Detection**: Accurate detection of license plates in various lighting and environmental conditions
- **Multi-class Support**: Extended support for detecting vehicles (car, motorcycle, bus), persons, and license plates
- **Training Pipeline**: Complete training workflow with data preprocessing
- **Testing & Validation**: Comprehensive testing suite with visualization capabilities
- **Real-time Processing**: Fast inference suitable for real-time applications

## 📁 Project Structure

```
├── README.md
├── train.py                    # Model training script
├── test.py                     # Model testing and visualization
├── foo.ipynb                   # Jupyter notebook for experimentation
├── data/
│   ├── process.py              # Data preprocessing utilities
│   └── archive/                # Original dataset
│       ├── classes.txt         # Class definitions
│       ├── dataset.yaml        # Dataset configuration
│       ├── images/
│       │   ├── train/          # Training images
│       │   └── val/            # Validation images
│       └── labels/
│           ├── train/          # Training labels (YOLO format)
│           └── val/            # Validation labels (YOLO format)
└── runs/                       # Training outputs (generated)
    └── detect/
        └── train3/
            └── weights/
                └── best.pt     # Best trained model weights
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/dvnam1605/Plate-Detection.git
cd "Plate detection"
```

2. Install required dependencies:
```bash
pip install ultralytics opencv-python matplotlib tqdm pyyaml
```

### Dataset Preparation

The dataset should follow YOLO format with:
- Images in `data/archive/images/train/` and `data/archive/images/val/`
- Corresponding labels in `data/archive/labels/train/` and `data/archive/labels/val/`
- Each label file contains bounding box coordinates in YOLO format: `class_id x_center y_center width height`

## 🎯 Usage

### Training the Model

To train the YOLOv11 model on your dataset:

```bash
python train.py
```

**Training Configuration:**
- **Epochs**: 100
- **Image Size**: 640x640
- **Batch Size**: 32
- **Learning Rate**: 5e-4
- **Data Augmentation**: Enabled
- **Training Time**: ~30 minutes on Tesla T4 GPU

**Expected Results:**
- License plate detection: >99% accuracy
- Multi-class object detection with good performance on cars and motorcycles
- Final model size: 19.2MB

### Testing the Model

To test the trained model and visualize results:

```bash
python test.py
```

This will:
- Load the best trained model from `runs/detect/train3/weights/best.pt`
- Run inference on random test images
- Display results with bounding boxes and confidence scores

### Data Processing

To preprocess and prepare the dataset:

```bash
python data/process.py
```

## 📊 Model Performance

### Training Results

**Training completed successfully in 0.497 hours (≈30 minutes)**

### Class-wise Performance

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|--------|-----------|-----------|---------|--------|----------|
| **All Classes** | 169 | 285 | 0.845 | 0.887 | 0.643 | 0.509 |
| **License Plate** | 169 | 169 | 0.981 | 0.994 | 0.995 | 0.863 |
| **Person** | 22 | 23 | 0.907 | 0.91 | 0.871 | 0.867 |
| **Car** | 129 | 131 | 0.886 | 0.813 | 0.886 | 0.817 |
| **Motorcycle** | 4 | 4 | 0.601 | 0.75 | 0.746 | 0.765 |
| **Bus** | 9 | 9 | 0.948 | 0.909 | 0.854 | 0.831 |

### Key Performance Highlights

- **License Plate Detection**: Excellent performance with 99.5% mAP50 and 86.3% mAP50-95
- **Car Detection**: Strong performance with 88.6% mAP50
- **Multi-class Support**: Successfully detects 5 different classes
- **Model Size**: Optimized 19.2MB model file
- **Inference Speed**: 0.2ms preprocessing, 12.6ms inference, 1.8ms postprocessing

### Model Architecture
- **Base Model**: YOLOv11s
- **Total Parameters**: 9,414,735
- **Model Layers**: 100 (fused)
- **GFLOPs**: 21.3
- **Hardware**: Tesla T4 GPU with CUDA support

## 🔍 Results

### Model Output Capabilities
- **Bounding boxes** around detected objects with high precision
- **Confidence scores** for each detection
- **Multi-class detection** with the following performance:
  - License plates: 99.5% accuracy (primary objective)
  - Cars: 88.6% accuracy
  - Motorcycles: 74.6% accuracy
  - Persons: 87.1% accuracy  
  - Buses: 85.4% accuracy

### Training Summary
- **Total Training Time**: ~30 minutes on Tesla T4 GPU
- **Model Size**: 19.2MB (highly optimized)
- **Validation Dataset**: 169 images with 285 object instances
- **Overall mAP50**: 64.3% across all classes
- **License Plate mAP50**: 99.5% (excellent primary performance)

### Real-time Performance
- **Preprocessing**: 0.2ms per image
- **Inference**: 12.6ms per image  
- **Postprocessing**: 1.8ms per image
- **Total Pipeline**: ~15ms per image (suitable for real-time applications)

## 📈 Model Optimization

Key training optimizations:
- Learning rate scheduling (lrf=0.1)
- Data augmentation enabled
- Adaptive batch sizing
- Early stopping capabilities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLOv11 implementation
- The open-source computer vision community
- Contributors to the license plate dataset

---

⭐ If you find this project helpful, please consider giving it a star!

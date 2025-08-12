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

Training parameters:
- **Epochs**: 100
- **Image Size**: 640x640
- **Batch Size**: 32
- **Learning Rate**: 5e-4
- **Data Augmentation**: Enabled

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

The model is trained to detect:
- **Primary Class**: License plates
- **Extended Classes**: Person, car, motorcycle, bus

Training configuration:
- Base Model: YOLOv11s
- Confidence Threshold: 0.25
- Multi-class detection support

## 🔍 Results

The trained model outputs:
- Bounding boxes around detected license plates
- Confidence scores for each detection
- Support for batch processing
- Real-time inference capabilities

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

## 📞 Contact

**Author**: dvnam1605  
**Repository**: [Plate-Detection](https://github.com/dvnam1605/Plate-Detection)

---

⭐ If you find this project helpful, please consider giving it a star!

# PBL
# Pothole Detection and Volume Estimation

This project implements an end-to-end pipeline for pothole detection, segmentation, and volume estimation using deep learning. The pipeline combines Mask R-CNN for pothole segmentation and MiDaS for depth estimation to calculate pothole volumes from video input.

## Features

- Pothole segmentation using Mask R-CNN (pretrained on COCO)
- Depth estimation using MiDaS
- Volume calculation in cubic centimeters
- Training pipeline for fine-tuning on custom datasets
- Video processing with visualization
- Easy-to-use Python API

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/pothole-detection.git
cd pothole-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Structure

Organize your dataset in the following structure:
```
data/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   └── ...
│   └── masks/
│       ├── image1.png
│       └── ...
└── val/
    ├── images/
    │   ├── image1.jpg
    │   └── ...
    └── masks/
        ├── image1.png
        └── ...
```

- Images should be in RGB format (JPG/JPEG/PNG)
- Masks should be binary images where white (255) represents pothole regions

## Usage

### Training

To train the model on your custom dataset:

```python
from pipeline import PotholePipeline

# Initialize pipeline
pipeline = PotholePipeline()

# Train model
pipeline.train_segmentation(
    train_dir="data/train",
    val_dir="data/val",
    num_epochs=10,
    batch_size=2
)

# Save trained model
pipeline.segmentation_model.save_checkpoint("models/segmentation_model.pth")
```

### Processing Video

To process a video and estimate pothole volumes:

```python
from pipeline import PotholePipeline

# Initialize pipeline with pretrained model
pipeline = PotholePipeline(
    segmentation_model_path="models/segmentation_model.pth",
    pixel_to_cm_ratio=0.1  # Adjust based on camera calibration
)

# Process video
pipeline.process_video(
    video_path="path/to/video.mp4",
    output_dir="results"
)
```

### Example Script

An example script is provided in `example.py`. Run it with:

```bash
python example.py
```

## Output

The pipeline generates the following outputs for each video frame:
- Segmentation masks with volume estimates
- Depth maps
- Overlay visualization
- Combined visualization saved as images

## Camera Calibration

The `pixel_to_cm_ratio` parameter is crucial for accurate volume estimation. To calibrate:
1. Place an object of known dimensions in the camera view
2. Measure the object's size in pixels
3. Calculate the ratio: size_in_cm / size_in_pixels

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 

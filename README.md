# Pedestrian Tracking and Re-identification Pipeline

## 1. Project Overview

This project provides a comprehensive pipeline for pedestrian tracking in video footage. It takes a video file as input, performs person detection using YOLOv5, and tracks individuals across frames using DeepSort. The system is designed to be modular and extensible, with components for preprocessing, tracking, feature extraction, and person re-identification model training.

This codebase has been significantly refactored from its original state to improve modularity, portability, and correctness.

## 2. Key Features

- **End-to-End Tracking Pipeline**: A single script (`main.py`) orchestrates the entire process from video to tracked frames.
- **Modular Design**: The project is broken down into logical modules for easy maintenance and extension.
- **Configurable Execution**: All scripts use command-line arguments, removing the need for hardcoded paths.
- **State-of-the-Art Models**: Utilizes YOLOv5 for person detection and DeepSort for tracking.
- **Feature Extraction**: Includes a module for extracting deep learning features from detected persons using a ResNet-50 model.
- **Person Re-ID Training**: Provides a script to train a person re-identification model (as a classification task).

## 3. Project Structure

```
.
├── Feature_Extraction/
│   └── feature_extractor.py   # Extracts deep learning features from images.
├── Person_reentry/
│   └── reentry.py             # Trains a person re-identification model.
├── Preprocessing/
│   └── Preprocessing.py       # Extracts and resizes frames from video.
├── Tracking/
│   └── tracking.py            # Performs person detection and tracking on frames.
├── output/
│   └──                        # Default directory for pipeline results.
├── main.py                    # Main script to run the entire pipeline.
├── requirements.txt           # Python dependencies.
└── yolov5s.pt                 # Pre-trained YOLOv5 model weights.
```

## 4. Setup and Installation

Follow these steps to set up the project environment.

**1. Clone the repository:**
```bash
git clone (https://github.com/Rupesh4604/Pedestrian-Tracking-using-Deep-Learning)
cd Pedestrian-Tracking-using-Deep-Learning
```

**2. Create a Python virtual environment (recommended):**
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies:**
Install all the required Python packages using the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

## 5. How to Run

### Running the Full Pipeline

The easiest way to use the project is to run the main pipeline script, which handles everything from frame extraction to tracking.

```bash
python3 main.py /path/to/your/video.mp4
```

This will create a new directory inside `output/` containing the extracted frames and the final frames with tracking overlays.

You can customize the run using optional arguments:
```bash
python3 main.py /path/to/your/video.mp4 --output_dir /path/to/custom_output --model yolov5m --conf 0.6
```
Use `python3 main.py --help` to see all available options.

### Running Individual Modules

You can also run each module independently, which is useful for debugging or specific tasks.

**1. Preprocessing:**
Extract frames from a video into a directory.
```bash
python3 Preprocessing/Preprocessing.py /path/to/video.mp4 /path/to/output_frames_dir
```

**2. Tracking:**
Run tracking on a directory of frames.
```bash
python3 Tracking/tracking.py /path/to/input_frames_dir /path/to/output_tracked_frames_dir
```

**3. Training the Re-ID Model:**
Train the person re-identification model using a dataset like `prid_450s`. The dataset should be organized with one subdirectory per person.
```bash
python3 Person_reentry/reentry.py /path/to/dataset_directory
```

## 6. Module Descriptions

- **`main.py`**: The central script that orchestrates the preprocessing and tracking pipeline.
- **`Preprocessing/Preprocessing.py`**: Contains functions to extract frames from a video file and resize them.
- **`Tracking/tracking.py`**: Uses YOLOv5 to detect people in frames and DeepSort to track them across the sequence of frames.
- **`Feature_Extraction/feature_extractor.py`**: Provides a `CNNFeatureExtractor` class that uses a pre-trained ResNet-50 to extract deep learning features from an image of a person. This is intended to be used for re-identification tasks.
- **`Person_reentry/reentry.py`**: A script for training a simple classification-based person re-identification model. It includes comments on its limitations and suggestions for more advanced metric learning approaches.

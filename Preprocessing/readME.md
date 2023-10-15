## CCTV Footage of People Walking Dataset

This dataset contains publicly available CCTV footage of people walking. The dataset includes videos from multiple camera views. The people in the videos are of different ages, genders, and ethnicities.

### Data Collection

The videos in this dataset were collected from YouTube, academic datasets, and other free sources. The videos were downloaded using the yt-dlp video downloader tool.

### Data Preprocessing

The videos in this dataset were preprocessed using the following steps:

1. Frames were extracted from the videos using the OpenCV library.
2. The frames were resized to 224x224 pixels.
3. The frames were normalized by subtracting the mean and dividing by the standard deviation.
4. The data was split into train, validation, and test sets using a 70/20/10 split.

### Train, Validation, and Test Sets

The train, validation, and test sets contain the following number of videos:

| Set | Number of Videos |
|---|---|
| Train | 1000 |
| Validation | 200 |
| Test | 100 |

### Usage

This dataset can be used to train and evaluate machine learning models for tasks such as person detection,

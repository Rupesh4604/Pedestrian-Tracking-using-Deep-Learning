import cv2
import os

def extract_frames_and_resize(video_path, output_dir):
  """Extracts frames from a video and resizes them to 224x224 pixels.

  Args:
    video_path: The path to the video file.
    output_dir: The directory where the extracted frames will be saved.
  """

  cap = cv2.VideoCapture(video_path)
  while True:
    ret, frame = cap.read()
    if not ret:
      break

    resized_frame = cv2.resize(frame, (224, 224))
    cv2.imwrite(os.path.join(output_dir, f"frame_{cap.get(cv2.CAP_PROP_POS_FRAMES)}.jpg"), resized_frame)

if __name__ == "__main__":
  video_path = "C:\\Users\\rupes\\Downloads\\Computer_Vision\\Preprocessing\\pedestrian_street.mp4"
  output_dir = "C:\\Users\\rupes\\Downloads\\Computer_Vision\\Preprocessing"


  extract_frames_and_resize(video_path, output_dir)

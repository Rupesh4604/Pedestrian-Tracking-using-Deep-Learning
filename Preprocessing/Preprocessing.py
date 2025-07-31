import cv2
import os
import argparse

def extract_frames(video_path, output_dir, resize_dim=(224, 224)):
  """Extracts frames from a video, resizes them, and saves them to an output directory.

  Args:
    video_path: The path to the video file.
    output_dir: The directory where the extracted frames will be saved.
    resize_dim: A tuple (width, height) to resize the frames.
  """
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    return

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  frame_count = 0
  while True:
    ret, frame = cap.read()
    if not ret:
      break

    resized_frame = cv2.resize(frame, resize_dim)
    output_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(output_path, resized_frame)
    frame_count += 1

  cap.release()
  print(f"Extracted {frame_count} frames from {video_path} to {output_dir}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Extract frames from a video.")
  parser.add_argument("video_path", type=str, help="Path to the video file.")
  parser.add_argument("output_dir", type=str, help="Directory to save the extracted frames.")
  parser.add_argument("--width", type=int, default=224, help="Width to resize frames.")
  parser.add_argument("--height", type=int, default=224, help="Height to resize frames.")
  args = parser.parse_args()

  resize_dim = (args.width, args.height)
  extract_frames(args.video_path, args.output_dir, resize_dim)

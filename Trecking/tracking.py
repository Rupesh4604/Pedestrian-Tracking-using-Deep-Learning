import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

def preprocess_image(image):
  """Pre-processes an image for YOLOv5.

  Args:
    image: A NumPy array representing the image.

  Returns:
    A PyTorch tensor representing the pre-processed image.
  """

  # Resize the image to 640x640 pixels.
  image = cv2.resize(image, (640, 640))

  # Convert the image to RGB format.
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  # Normalize the image to a range of [0, 1].
  image = image / 255.0

  # Transpose the image to (channels, height, width) format.
  image = image.transpose((2, 0, 1))

  # Convert the image to a PyTorch tensor.
  image = torch.from_numpy(image)

  return image

# Load the YOLOv5 model.
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

# Create the tracker.
tracker = DeepSort(max_age=5)

# Load the input image.
image = cv2.imread("C:\\Users\\rupes\\Downloads\\Computer_Vision\\Preprocessing\\frame_4.0.jpg")

# Pre-process the input image.
preprocessed_image = preprocess_image(image)

# Detect people in the input image.
results = model(preprocessed_image.unsqueeze(0)).cpu().detach().pandas().xyxy[0]

# Filter the bounding boxes.
bounding_boxes = []
predicted_classes = []
for row in results.itertuples():
  if row.confidence > 0.5 and row.name == "person":
    bounding_boxes.append([row.xmin, row.ymin, row.xmax, row.ymax])
    predicted_classes.append(row.name)

# Update the tracker with the new detections.
tracks = tracker.update_tracks(bounding_boxes, frame=image)

# Draw bounding boxes around the tracked people.
for track in tracks:
  if not track.is_confirmed():
    continue

  track_id = track.track_id
  ltrb = track.to_ltrb()

  cv2.rectangle(image, (ltrb[0], ltvb[1]), (ltrb[2], ltvb[3]), (0, 255, 0), 2)
  cv2.putText(image, str(track_id), (ltrb[0], ltvb[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the output image.
cv2.imshow("Output", image)
cv2.waitKey(0)

# Release the video capture.
cv2.destroyAllWindows()

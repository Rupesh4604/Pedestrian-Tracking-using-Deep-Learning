import cv2
import torch

from deep_sort import nn_matching
from deep_sort import preprocessing
from deep_sort import detection
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

class PersonDetector:
  def __init__(self, model_path):
    self.model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

  def detect_people(self, image):
    """Detects people in an image.

    Args:
      image: A numpy array representing the image.

    Returns:
      A list of bounding boxes and predicted classes for each bounding box.
    """

    # Preprocess the image.
    image = cv2.resize(image, (640, 640))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose((2, 0, 1))
    image = image / 255.0
    image = image[:, :, ::-1]

    # Apply the model to the image.
    results = self.model(image.unsqueeze(0)).pandas().xyxy[0]

    # Filter the bounding boxes.
    bounding_boxes = []
    predicted_classes = []
    for row in results.itertuples():
      if row.confidence > 0.5 and row.name == "person":
        bounding_boxes.append([row.xmin, row.ymin, row.xmax, row.ymax])
        predicted_classes.append(row.name)

    return bounding_boxes, predicted_classes

class PersonTracker:
  def __init__(self):
    self.tracker = Tracker(nn_matching.NearestNeighborDistanceMetric("cosine"), max_age=30, min_hits=3)

  def track_people(self, image, bounding_boxes, predicted_classes):
    """Tracks people in an image.

    Args:
      image: A numpy array representing the image.
      bounding_boxes: A list of bounding boxes.
      predicted_classes: A list of predicted classes for each bounding box.

    Returns:
      A list of bounding boxes and track IDs for each individual in the image.
    """

    # Preprocess the image.
    image = preprocessing.preprocess_image(image)

    # Create a list of detections.
    detections = []
    for i in range(len(bounding_boxes)):
      detection = Detection(bounding_boxes[i], predicted_classes[i])
      detections.append(detection)

    # Update the tracker.
    self.tracker.predict()
    self.tracker.update(detections)

    # Get the tracking results.
    tracking_results = self.tracker.get_results()

    # Draw the bounding boxes on the image.
    for tracking_result in tracking_results:
      bounding_box = tracking_result.to_tlbr()
      track_id = tracking_result.track_id
      cv2.rectangle(image, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), (0, 255, 0), 2)
      cv2.putText(image, str(track_id), (bounding_box[0], bounding_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image, tracking_results

if __name__ == "__main__":
  image_path = "C:\\Users\\rupes\\Downloads\\Computer_Vision\\Preprocessing\\pedestrian_street.mp4"

  # Load the pre-trained person detector model.
  person_detector = PersonDetector()

  # Detect people in the image.
  bounding_boxes, predicted_classes = person_detector.detect_people(cv2.imread(image_path))

  # Create a person tracker.
  person_tracker = PersonTracker()

  # Track people in the image.
  image, tracking_results = person_tracker.track_people(image_path, bounding_boxes, predicted_classes)

  # Display the image.
  cv2.imshow("Image", image)
  cv2.waitKey(0)

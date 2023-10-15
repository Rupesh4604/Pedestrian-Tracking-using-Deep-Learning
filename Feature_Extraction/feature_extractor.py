import torch
import torchvision.models as models

class CNNFeatureExtractor:
  def __init__(self, model_path):
    self.model = models.resnet50(pretrained=True)

  def extract_features(self, image):
    """Extracts CNN features from an image.

    Args:
      image: A numpy array representing the image.

    Returns:
      A numpy array representing the CNN features.
    """

    # Preprocess the image.
    image = cv2.resize(image, (224, 224))
    image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

    # Extract the features.
    features = self.model(image.unsqueeze(0))

    return features

# Create a CNN feature extractor.
cnn_feature_extractor = CNNFeatureExtractor("path/to/model.pt")

# Extract CNN features from the detected and tracked individuals.
features = []
for bounding_box in bounding_boxes:
  image = image[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2]]
  features.append(cnn_feature_extractor.extract_features(image))

# Save the features to a file.
with open("features.txt", "w") as f:
  for feature in features:
    f.write(str(feature) + "\n")

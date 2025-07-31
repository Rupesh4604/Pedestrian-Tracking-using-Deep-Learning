import torch
import torchvision.models as models
import cv2
import numpy as np
from torchvision import transforms

class CNNFeatureExtractor:
  def __init__(self):
    """
    Initializes the feature extractor with a pre-trained ResNet-50 model.
    """
    self.model = models.resnet50(pretrained=True)
    self.model.eval()  # Set the model to evaluation mode

    # Define the image transformations
    self.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

  def extract_features(self, image):
    """
    Extracts CNN features from an image.

    Args:
      image: A numpy array representing the image (in BGR format from OpenCV).

    Returns:
      A numpy array representing the CNN features.
    """
    # Convert image from BGR to RGB, resize, and apply transformations
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (224, 224))

    # Apply the transformations
    input_tensor = self.transform(image_resized)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

    # Extract features
    with torch.no_grad():
        features = self.model(input_batch)

    return features.cpu().numpy()

if __name__ == '__main__':
    # Example usage (optional, for testing)
    # Create a dummy image
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Initialize the extractor
    feature_extractor = CNNFeatureExtractor()

    # Extract features
    features = feature_extractor.extract_features(dummy_image)

    print("Feature extraction example:")
    print(f"Input image shape: {dummy_image.shape}")
    print(f"Extracted features shape: {features.shape}")
    # The output of ResNet50 before the final FC layer is (1, 2048, 1, 1)
    # but since we are using the model with its final layer, it's (1, 1000)
    # for ImageNet classes. This is a good feature vector.

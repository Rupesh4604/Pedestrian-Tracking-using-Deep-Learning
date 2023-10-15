import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Define the model architecture
class PersonReIDModel(nn.Module):
    def __init__(self):
        super(PersonReIDModel, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )
        # Adjust the input size to match the expected input size
        self.fc = nn.Linear(64 * 16 * 16, 128)  # Adjust the input size based on feature map size
        self.relu = nn.ReLU()
        self.output = nn.Linear(128, 64)

    def forward(self, x):
        features = self.network(x)
        features = self.fc(features)
        features = self.relu(features)
        features = self.output(features)
        return features

# Define a function to load the training data
def load_training_data(data_dir):
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    train_dataset = datasets.ImageFolder(data_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    return train_loader

# Define a function to evaluate the model
def evaluate_model(model, test_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Define the path to the "prid_450s" dataset directory
data_dir = "C:\\Users\\rupes\\Downloads\\prid_450s"

# Create the model
model = PersonReIDModel()

# Load training data
train_loader = load_training_data(data_dir)

# Define the loss function (Cross-Entropy)
criterion = nn.CrossEntropyLoss()

# Define the optimizer (e.g., Adam)
optimizer = optim.Adam(model.parameters())

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        features = model(images)
        loss = criterion(features, labels)
        loss.backward()
        optimizer.step()

# Evaluate the model
# You can create a separate test loader using the same data directory.
test_loader = load_training_data(data_dir)
accuracy = evaluate_model(model, test_loader, criterion)

print("Accuracy:", accuracy)

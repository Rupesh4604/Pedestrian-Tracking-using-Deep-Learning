import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse

# --- Model Definition ---
# NOTE: This model treats person re-identification as a classification problem,
# where each person ID is a class. This is a valid approach for "closed-set"
# re-id, where all test identities are known during training.
# A more common and powerful approach is metric learning (e.g., with Triplet Loss),
# which learns an embedding space where images of the same person are close and
# images of different people are far apart. This allows for "open-set" re-id.

class PersonReIDModel(nn.Module):
    def __init__(self, num_classes):
        super(PersonReIDModel, self).__init__()
        self.network = nn.Sequential(
            # Input: (3, 64, 64)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> (32, 32, 32)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> (64, 16, 16)
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes) # Output layer with `num_classes`
        )

    def forward(self, x):
        return self.network(x)

# --- Data Loading ---
def get_data_loaders(data_dir, batch_size, val_split=0.2):
    """
    Loads the dataset and splits it into training and validation loaders.
    """
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(data_dir, transform=transform)

    # --- Data Splitting ---
    # It's crucial to evaluate the model on a separate validation/test set
    # that it hasn't seen during training.
    num_train = int((1.0 - val_split) * len(full_dataset))
    num_val = len(full_dataset) - num_train
    train_dataset, val_dataset = random_split(full_dataset, [num_train, num_val])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_classes = len(full_dataset.classes)
    print(f"Dataset has {num_classes} classes (person identities).")

    return train_loader, val_loader, num_classes

# --- Evaluation Function ---
def evaluate_model(model, data_loader, criterion, device):
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(data_loader)
    return avg_loss, accuracy

# --- Main Training and Evaluation Logic ---
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_loader, val_loader, num_classes = get_data_loaders(args.data_dir, args.batch_size)

    # Create the model
    model = PersonReIDModel(num_classes).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train() # Set model to training mode
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # --- Validation ---
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{args.epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    print("Training finished.")

    # Final evaluation on the validation set
    final_loss, final_accuracy = evaluate_model(model, val_loader, criterion, device)
    print(f"Final Validation Accuracy: {final_accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Person Re-identification Model.")
    parser.add_argument("data_dir", type=str, help="Path to the dataset directory (e.g., prid_450s).")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizer.")

    args = parser.parse_args()
    main(args)

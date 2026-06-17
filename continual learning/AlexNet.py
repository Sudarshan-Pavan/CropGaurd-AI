import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import time

# Renate imports
import renate.defaults as defaults
from renate.benchmark.datasets.vision_datasets import TorchVisionDataModule
from renate.benchmark.scenarios import ClassIncrementalScenario
from renate.models import RenateModule

# User inputs for hyperparameters
try:
    batch_size = int(input("Enter batch size: "))
    num_epochs = int(input("Enter number of epochs: "))
    early_stop_patience = int(input("Enter early stopping patience: "))
    learning_rate = float(input("Enter learning rate: "))
    dropout_rate = float(input("Enter dropout rate: "))
    print("Inputs received successfully.")
except Exception as e:
    print("Error in user input:", e)
    exit()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
dataset_path = r"C:\stse\capstone\a\small dataset to test\CICIDS_Multi"
try:
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    print(f"Dataset loaded successfully with {len(dataset)} samples.")
except Exception as e:
    print("Error loading dataset:", e)
    exit()

# Split dataset into train, val, and test
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
print(f"Dataset split into train ({train_size}), val ({val_size}), and test ({test_size}) sets.")

# Dataloaders
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Define a custom model class that inherits from RenateModule
class MyRenateModel(RenateModule):
    def __init__(self, base_model):
        super().__init__(constructor_arguments={})
        self.base_model = base_model

    def forward(self, x):
        return self.base_model(x)

# Load pre-trained AlexNet model and wrap it in your custom RenateModule
model = models.alexnet(weights="DEFAULT")
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Sequential(
    nn.Dropout(dropout_rate),
    nn.Linear(num_features, len(dataset.classes))
)

# Use the custom model with Renate's module
model = MyRenateModel(model)
print("Model initialized.")

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Set up the continual learning scenario with Renate
data_module = TorchVisionDataModule(
    dataset_path, dataset_name="CIFAR10", val_size=0.2, seed=defaults.SEED
)
scenario = ClassIncrementalScenario(
    data_module=data_module,
    groupings=((0, 1, 2, 3, 4), (5, 6, 7, 8, 9)),
    chunk_id=0
)
print("Class Incremental Scenario initialized.")

# Training loop with early stopping
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
best_val_loss = float('inf')
no_improve_count = 0
best_model = None
start_time = time.time()

for epoch in range(num_epochs):
    print(f"Starting epoch {epoch + 1}/{num_epochs}")
    model.train()  # Set model to training mode

    running_loss = 0.0
    correct_preds = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct_preds += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_set)
    train_acc = correct_preds / len(train_set)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct_preds = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct_preds += (predicted == labels).sum().item()

    val_loss /= len(val_set)
    val_acc = val_correct_preds / len(val_set)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(
        f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

    # Early stopping logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
        no_improve_count = 0
    else:
        no_improve_count += 1
        if no_improve_count >= early_stop_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

end_time = time.time()

# Load best model for final evaluation
model = best_model
model.eval()
test_correct_preds = 0
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        test_correct_preds += (predicted == labels).sum().item()

test_accuracy = accuracy_score(all_labels, all_preds)
test_f1 = f1_score(all_labels, all_preds, average='weighted')
test_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
test_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=1)

# Print final metrics
print(
    f"Test Accuracy: {test_accuracy:.4f}, F1-Score: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
print(f"Training completed in {end_time - start_time:.2f} seconds")

# Plot loss vs accuracy curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Loss over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Val Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.show()

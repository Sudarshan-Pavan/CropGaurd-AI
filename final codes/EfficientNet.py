import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, roc_curve, auc
import time
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings

warnings.filterwarnings('ignore')

# Define transformations for the dataset
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load features CSV and encode labels
features_df = pd.read_csv('paddy_doctor_glcm_features.csv')
le = LabelEncoder()
features_df['label'] = le.fit_transform(features_df['image_path'].apply(
    lambda x: os.path.basename(os.path.dirname(x))))

# Split the dataset into train, validation, and test
train_df, temp_df = train_test_split(
    features_df, test_size=0.3, random_state=42, stratify=features_df['label'])
val_df, test_df = train_test_split(
    temp_df, test_size=0.66, random_state=42, stratify=temp_df['label'])

# Check the number of images in each set
print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")


# Custom dataset class
class CustomImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image_path']
        image = Image.open(img_path).convert("RGB")
        label = self.dataframe.iloc[idx]['label']

        if self.transform:
            image = self.transform(image)

        return image, label


# Create dataset objects for training, validation, and test
train_dataset = CustomImageDataset(
    train_df, transform=data_transforms['train'])
val_dataset = CustomImageDataset(val_df, transform=data_transforms['val'])
test_dataset = CustomImageDataset(test_df, transform=data_transforms['test'])

# User inputs for hyperparameters
batch_size = int(input("Enter batch size: "))
epochs = int(input("Enter number of epochs: "))
earlystop_patience = int(input("Enter early stop patience: "))
learning_rate = float(input("Enter learning rate: "))
dropout_rate = float(input("Enter dropout rate: "))

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define model, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub',
                       'nvidia_efficientnet_b0', pretrained=True)

# Modify the model for the number of classes in your dataset
num_classes = len(features_df['label'].unique())
model.classifier.fc = nn.Sequential(
    nn.Dropout(dropout_rate),
    nn.Linear(model.classifier.fc.in_features, num_classes)
)

model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Early stopping and scheduler
scheduler = ReduceLROnPlateau(
    optimizer, mode='min', patience=earlystop_patience, verbose=True)


class EarlyStopping:
    def __init__(self, patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


early_stop_counter = 0
best_val_loss = float('inf')

# Create output directory
output_dir = "output_EfficientNet"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to calculate evaluation metrics


def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy, all_preds, all_labels


# Training loop
train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

for epoch in range(epochs):
    model.train()
    running_loss, correct_preds = 0.0, 0
    start_time = time.time()

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_preds += torch.sum(preds == labels.data)

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = correct_preds.double() / len(train_loader.dataset)
    val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc.item())
    val_accuracies.append(val_acc)

    scheduler.step(val_loss)

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), os.path.join(
            output_dir, "best_model.pth"))
    else:
        early_stop_counter += 1
        if early_stop_counter >= earlystop_patience:
            print("Early stopping")
            break

    print(f"Epoch {epoch + 1}/{epochs} - Time: {time.time() - start_time:.2f}s")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

    # Save performance metrics to a CSV file
    epoch_performance = {
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_accuracy': train_acc.item(),
        'val_accuracy': val_acc
    }
    performance_df = pd.DataFrame([epoch_performance])
    performance_file = os.path.join(output_dir, "epoch_performance.csv")
    if os.path.exists(performance_file):
        performance_df.to_csv(performance_file, mode='a',
                              header=False, index=False)
    else:
        performance_df.to_csv(performance_file, index=False)

# Load best model
model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pth")))

# Evaluate on the test set
test_loss, test_acc, test_preds, test_labels = evaluate(
    model, test_loader, criterion)

# Print final evaluation metrics
precision = precision_score(test_labels, test_preds, average='weighted')
recall = recall_score(test_labels, test_preds, average='weighted')
f1 = f1_score(test_labels, test_preds, average='weighted')
fpr, tpr, _ = roc_curve(test_labels, test_preds, pos_label=1)
roc_auc = auc(fpr, tpr)

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
print(f"AUC: {roc_auc:.4f}")

# Save final evaluation metrics to a file
final_metrics = {
    'test_loss': test_loss,
    'test_accuracy': test_acc,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'roc_auc': roc_auc
}
final_metrics_df = pd.DataFrame([final_metrics])
final_metrics_file = os.path.join(output_dir, "final_metrics.csv")
final_metrics_df.to_csv(final_metrics_file, index=False)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss per Epoch')
plt.savefig(os.path.join(output_dir, 'loss_per_epoch.png'))

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy per Epoch')
plt.savefig(os.path.join(output_dir, 'accuracy_per_epoch.png'))

plt.show()

# Save box plots for performance metrics
plt.figure(figsize=(12, 6))
plt.boxplot([train_losses, val_losses, train_accuracies, val_accuracies],
            labels=['Train Loss', 'Val Loss', 'Train Accuracy', 'Val Accuracy'])
plt.title('Performance Metrics Box Plot')
plt.savefig(os.path.join(output_dir, 'performance_box_plot.png'))

plt.show()

# Save the best model
torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))

print(f"All outputs have been saved to the folder '{output_dir}'")

# Ensure you log these print statements to a log file as well
log_file = os.path.join(output_dir, 'training_log.txt')
with open(log_file, 'w') as f:
    f.write(f"Training samples: {len(train_df)}\n")
    f.write(f"Validation samples: {len(val_df)}\n")
    f.write(f"Test samples: {len(test_df)}\n")
    for epoch in range(len(train_losses)):
        f.write(f"Epoch {epoch + 1}/{epochs}\n")
        f.write(
            f"Train Loss: {train_losses[epoch]:.4f}, Train Accuracy: {train_accuracies[epoch]:.4f}\n")
        f.write(
            f"Val Loss: {val_losses[epoch]:.4f}, Val Accuracy: {val_accuracies[epoch]:.4f}\n")
    f.write(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}\n")
    f.write(
        f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}\n")
    f.write(f"AUC: {roc_auc:.4f}\n")

print(f"Training log saved to '{log_file}'")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, accuracy_score, auc
from torch.optim.lr_scheduler import StepLR
import time
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image


# Load and prepare the dataset
features_df = pd.read_csv(r'paddy_doctor_glcm_features.csv')

# Encode labels
le = LabelEncoder()
features_df['label'] = le.fit_transform(features_df['image_path'].apply(
    lambda x: os.path.basename(os.path.dirname(x))))

# Split the dataset into train, validation, and test
train_df, temp_df = train_test_split(
    features_df, test_size=0.3, random_state=42, stratify=features_df['label'])
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

# Check the number of images in each set
print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")

# User inputs
batch_size = int(input("Enter the batch size: "))
num_epochs = int(input("Enter the number of epochs: "))
earlystop_patience = int(input("Enter early stop patience: "))
learning_rate = float(input("Enter the learning rate: "))
dropout_rate = float(input("Enter the dropout rate: "))

# Define the custom dataset


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


# Dataset transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize from 480x640 to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Create datasets
train_dataset = CustomImageDataset(train_df, transform=transform)
val_dataset = CustomImageDataset(val_df, transform=transform)
test_dataset = CustomImageDataset(test_df, transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load the pre-trained ShuffleNet model
model = models.shufflenet_v2_x1_0(weights="DEFAULT")
model.eval()

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Modify the classifier part of the model (since we have different classes)
# Replace with actual number of classes in the dataset
num_classes = len(le.classes_)
model.fc = nn.Sequential(
    nn.Dropout(p=dropout_rate),  # Dropout layer
    nn.Linear(model.fc.in_features, num_classes)
).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

# Early stopping


class EarlyStopping:
    def __init__(self, patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model = model.state_dict()
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.best_model = model.state_dict()

# Track evaluation metrics


def compute_metrics(y_true, y_pred, y_probs):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    auc_score = roc_auc_score(y_true, y_probs, multi_class='ovr')
    precision, recall, _ = precision_recall_curve(
        y_true, y_probs[:, 1], pos_label=1)
    auc_pr = auc(recall, precision)
    return acc, f1, auc_score, precision, recall, auc_pr


# Create output directory
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Training loop
early_stopping = EarlyStopping(patience=earlystop_patience, verbose=True)

train_loss_values, val_loss_values = [], []
train_acc_values, val_acc_values = [], []
epoch_times = []

for epoch in range(num_epochs):
    start_time = time.time()

    model.train()
    train_loss, correct_train = 0, 0
    total_train = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_train += torch.sum(preds == labels.data)
        total_train += labels.size(0)

    train_loss = train_loss / total_train
    train_accuracy = correct_train.double() / total_train
    train_loss_values.append(train_loss)
    train_acc_values.append(train_accuracy.item())

    # Validation
    model.eval()
    val_loss, correct_val = 0, 0
    total_val = 0
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_val += torch.sum(preds == labels.data)
            total_val += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    val_loss = val_loss / total_val
    val_accuracy = correct_val.double() / total_val
    val_loss_values.append(val_loss)
    val_acc_values.append(val_accuracy.item())

    # Compute additional metrics
    acc, f1, auc_score, precision, recall, auc_pr = compute_metrics(
        all_labels, all_preds, np.array(all_probs))

    epoch_time = time.time() - start_time
    epoch_times.append(epoch_time)

    print(f'Epoch {epoch+1}/{num_epochs}, Time: {epoch_time:.2f}s')
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}')
    print(
        f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, F1: {f1:.4f}, AUC: {auc_score:.4f}, AUC PR: {auc_pr:.4f}')

    # Early stopping
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

    scheduler.step()

# Save the best model
best_model_path = os.path.join(output_dir, 'best_model.pth')
torch.save(early_stopping.best_model, best_model_path)

# Save training and validation metrics
metrics = pd.DataFrame({
    'Epoch': range(1, len(train_loss_values) + 1),
    'Train Loss': train_loss_values,
    'Val Loss': val_loss_values,
    'Train Accuracy': train_acc_values,
    'Val Accuracy': val_acc_values,
    'Time per Epoch': epoch_times
})
metrics.to_csv(os.path.join(output_dir, 'training_metrics.csv'), index=False)

# Plot training & validation loss and accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss_values, label='Train Loss')
plt.plot(val_loss_values, label='Validation Loss')
plt.title('Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(output_dir, 'loss_vs_epoch.png'))

plt.subplot(1, 2, 2)
plt.plot(train_acc_values, label='Train Accuracy')
plt.plot(val_acc_values, label='Validation Accuracy')
plt.title('Accuracy vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(output_dir, 'accuracy_vs_epoch.png'))

plt.show()

# Print epoch times
print(f"Average training time per epoch: {np.mean(epoch_times):.2f}s")

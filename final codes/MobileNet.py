import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

# Create output folder
output_folder = "output_MobileNet"
os.makedirs(output_folder, exist_ok=True)

# User inputs for hyperparameters
batch_size = int(input("Enter batch size: "))
num_epochs = int(input("Enter number of epochs: "))
early_stop_patience = int(input("Enter early stopping patience: "))
learning_rate = float(input("Enter learning rate: "))
dropout_rate = float(input("Enter dropout rate: "))

# Data transformations for images
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
    ])
}

# Load features CSV and encode labels
features_df = pd.read_csv('paddy_doctor_glcm_features.csv')
le = LabelEncoder()
features_df['label'] = le.fit_transform(features_df['image_path'].apply(
    lambda x: os.path.basename(os.path.dirname(x))))

# Split the dataset into train, validation, and test sets
train_df, temp_df = train_test_split(
    features_df, test_size=0.3, random_state=42, stratify=features_df['label'])
val_df, test_df = train_test_split(
    temp_df, test_size=0.66, random_state=42, stratify=temp_df['label'])

print(
    f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}, Test samples: {len(test_df)}")

# Custom Dataset class


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


# Create dataset instances and DataLoader objects
train_dataset = CustomImageDataset(
    train_df, transform=data_transforms['train'])
val_dataset = CustomImageDataset(val_df, transform=data_transforms['val'])
test_dataset = CustomImageDataset(test_df, transform=data_transforms['test'])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the model with MobileNetV2


class ModifiedMobileNetV2(nn.Module):
    def __init__(self, dropout_rate):
        super().__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.model.classifier[1].in_features, len(le.classes_))
        )

    def forward(self, x):
        return self.model(x)


# Instantiate the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ModifiedMobileNetV2(dropout_rate=dropout_rate).to(device)

# Loss function, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=3, factor=0.1)


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


# Initialize variables for saving the best model
best_val_loss = float('inf')
best_model_path = os.path.join(output_folder, "best_model.pth")

# Training and validation


def train_and_validate():
    global best_val_loss
    early_stop_counter = 0
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    performance_log = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss, correct_train, total_train = 0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            _, preds = outputs.max(1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_loss = total_train_loss / len(train_loader)
        train_acc = correct_train / total_train * 100

        # Validation
        model.eval()
        total_val_loss, correct_val, total_val = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                _, preds = outputs.max(1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_loss = total_val_loss / len(val_loader)
        val_acc = correct_val / total_val * 100

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        performance_log.append({'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss,
                                'train_acc': train_acc, 'val_acc': val_acc})

        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print("Early stopping triggered.")
                break

        scheduler.step(val_loss)

    # Save performance log as a CSV file
    pd.DataFrame(performance_log).to_csv(os.path.join(
        output_folder, "performance_log.csv"), index=False)

    return train_losses, val_losses, train_accuracies, val_accuracies


train_losses, val_losses, train_accuracies, val_accuracies = train_and_validate()

# Plot the learning curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Loss')
plt.legend()
plt.savefig(os.path.join(output_folder, "loss_curve.png"))

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Val Accuracy')
plt.title('Accuracy')
plt.legend()
plt.savefig(os.path.join(output_folder, "accuracy_curve.png"))

plt.show()

# Model evaluation on the test set


def evaluate_model():
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    report = classification_report(
        all_labels, all_preds, target_names=le.classes_, output_dict=True)
    print(f"Test Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    print("Classification Report:\n", classification_report(
        all_labels, all_preds, target_names=le.classes_))

    # Save classification report and performance metrics
    pd.DataFrame(report).transpose().to_csv(
        os.path.join(output_folder, "classification_report.csv"))


evaluate_model()

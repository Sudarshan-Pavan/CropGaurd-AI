import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, accuracy_score, auc
import matplotlib.pyplot as plt
import numpy as np
from renate.updaters.experimental.fine_tuning import FineTuningModelUpdater
from renate.models.renate_module import RenateWrapper

# Load the pre-trained ShuffleNet model
model = models.shufflenet_v2_x1_0(weights="DEFAULT")
model.eval()

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# User inputs
batch_size = int(input("Enter the batch size: "))
num_epochs = int(input("Enter the number of epochs: "))
earlystop_patience = int(input("Enter early stop patience: "))
learning_rate = float(input("Enter the learning rate: "))
dropout_rate = float(input("Enter the dropout rate: "))

# Dataset transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load dataset
dataset = datasets.ImageFolder(root='/kaggle/input/cicids-multi-16k/CICIDS_Multi_16k', transform=transform)

# Split dataset into training (70%), validation (15%), and test (15%)
train_size = int(0.5 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Further split the train dataset into three batches
batch_1_size = len(train_dataset) // 3
batch_2_size = (len(train_dataset) - batch_1_size) // 2
batch_3_size = len(train_dataset) - batch_1_size - batch_2_size

train_batch_1, train_batch_2, train_batch_3 = random_split(train_dataset, [batch_1_size, batch_2_size, batch_3_size])

# Create data loaders for each batch
train_loader_1 = DataLoader(train_batch_1, batch_size=batch_size, shuffle=True)
train_loader_2 = DataLoader(train_batch_2, batch_size=batch_size, shuffle=True)
train_loader_3 = DataLoader(train_batch_3, batch_size=batch_size, shuffle=True)

# List of train loaders for looping
train_loaders = [train_loader_1, train_loader_2, train_loader_3]

# Modify the classifier part of the model (since we have different classes)
num_classes = len(dataset.classes)  # Number of classes in the dataset
model.fc = nn.Sequential(
    nn.Dropout(p=dropout_rate),  # Use the dropout rate from user input
    nn.Linear(model.fc.in_features, num_classes)
)

# Dataloaders for validation and test sets
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Renate: Wrap the model and define the loss function
model = RenateWrapper(model)
criterion = nn.CrossEntropyLoss(reduction="mean")

# Use an optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Early stopping implementation
class EarlyStopping:
    def _init_(self, patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def _call_(self, val_loss):
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

# Track evaluation metrics
def compute_metrics(y_true, y_pred, y_probs):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    auc_score = roc_auc_score(y_true, y_probs, multi_class='ovr')
    precision, recall, _ = precision_recall_curve(y_true, y_probs[:, 1], pos_label=1)
    auc_pr = auc(recall, precision)
    return acc, f1, auc_score, precision, recall, auc_pr

# Instantiate the ModelUpdater for training
updater = FineTuningModelUpdater(
    model,
    criterion,
    optimizer=optimizer,
    batch_size=batch_size,
    max_epochs=num_epochs,
    output_state_folder="renate_output",
)

# Start training using Renate with early stopping
early_stopping = EarlyStopping(patience=earlystop_patience, verbose=True)

for epoch in range(num_epochs):
    for batch_idx, train_loader in enumerate(train_loaders):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        print(f"Training on batch {batch_idx + 1} of 3 for epoch {epoch + 1}")

        # Training phase for current batch
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()   # Reset gradients
            outputs = model(inputs) # Forward pass
            loss = criterion(outputs, labels) # Compute loss
            loss.backward()         # Backward pass
            optimizer.step()        # Update weights

            running_loss += loss.item()
            
            # Calculate training accuracy
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data)
            total_train += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        train_accuracy = correct_train.double() / total_train
        print(f'Batch {batch_idx + 1} Loss: {epoch_loss:.4f}, Train Acc: {train_accuracy:.4f}')

    # Validation phase after all batches
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # Calculate validation accuracy
            _, preds = torch.max(outputs, 1)
            correct_val += torch.sum(preds == labels.data)
            total_val += labels.size(0)

    val_loss /= len(val_loader)
    val_accuracy = correct_val.double() / total_val
    print(f'Validation Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

    # Early stopping
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered.")
        break

# Test the model
model.eval()
test_loss, correct_test = 0, 0
total_test = 0
all_test_labels, all_test_preds, all_test_probs = [], [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        test_loss += loss.sum().item()  # Change here to sum the loss
        _, preds = torch.max(outputs, 1)
        correct_test += torch.sum(preds == labels.data)
        total_test += labels.size(0)

        all_test_labels.extend(labels.cpu().numpy())
        all_test_preds.extend(preds.cpu().numpy())
        all_test_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

test_loss = test_loss / total_test
test_accuracy = correct_test.double() / total_test

# Compute test metrics
test_acc, test_f1, test_auc, precision, recall, auc_pr = compute_metrics(all_test_labels, all_test_preds, np.array(all_test_probs))

print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}')
print(f'Test F1: {test_f1:.4f}, Test AUC: {test_auc:.4f}, Test AUC PR: {auc_pr:.4f}')

# Plot Precision-Recall curve
plt.figure()
plt.plot(recall, precision, label=f'Precision-Recall AUC: {auc_pr:.4f}')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()
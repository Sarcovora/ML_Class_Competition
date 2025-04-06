import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import wandb
import os

# Load data
data_path = os.path.join('data', 'processed_train_data.pkl')
train_data = pd.read_pickle(data_path)
print("Processed dataset loaded.")

train_data = train_data.drop(columns=['Breed', 'Color'])

train_x = train_data.drop('Outcome Type', axis=1)
train_y = train_data['Outcome Type']

outcome_mapping = {
	'Return to Owner': 0,
	'Transfer': 1,
	'Adoption': 2,
	'Died': 3,
	'Euthanasia': 4
}

train_y_encoded = train_y.map(outcome_mapping)

X_np = train_x.values.astype(np.float32)
y_np = train_y_encoded.values.astype(np.int64)

X_tensor = torch.tensor(X_np)
y_tensor = torch.tensor(y_np)

# Create a TensorDataset and split 10% for validation
dataset = TensorDataset(X_tensor, y_tensor)
val_size = int(0.1 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


# Get the number of input features (columns in train_x) and set hidden layer size and output classes.
input_dim = X_tensor.shape[1]
hidden_dim = 64   # you can adjust this hyperparameter
output_dim = 5    # 5 outcome classes

# Initialize the model.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNet(input_dim, hidden_dim, output_dim).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize wandb project
wandb.init(entity='evan-ekuo-edu', project="ML Course Project", config={
    "epochs": 20,
    "batch_size": batch_size,
    "learning_rate": 0.001,
    "hidden_dim": hidden_dim
})
num_epochs = wandb.config.epochs

# ----- Training Loop -----
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    # Wrap the dataloader with tqdm for a progress bar.
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    for inputs, labels in progress_bar:
        # Move data to GPU if available.
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        progress_bar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / train_size
    wandb.log({"epoch": epoch+1, "loss": epoch_loss})
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')


torch.save(model.state_dict(), "model_weights.pth")
print("Model weights saved to model_weights.pth")

# -------------------------
# Evaluation on the Validation Set
# -------------------------
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Validation Accuracy: {accuracy*100:.2f}%')
wandb.log({"validation_accuracy": accuracy})
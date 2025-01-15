import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib

# Load the dataset from train.csv
data = pd.read_csv('train.csv')

# Feature Engineering
le_state = LabelEncoder()
le_crop = LabelEncoder()
le_soil = LabelEncoder()
data['State'] = le_state.fit_transform(data['State'])
data['Crop_Type'] = le_crop.fit_transform(data['Crop_Type'])
data['Soil_Type'] = le_soil.fit_transform(data['Soil_Type'])

# Input and output
X = data.drop(columns=['Crop_Yield (kg/ha)', 'id'])
y = data['Crop_Yield (kg/ha)']

# Normalize numeric features
scaler = StandardScaler()
X[['Rainfall', 'Irrigation_Area','Year']] = scaler.fit_transform(X[['Rainfall', 'Irrigation_Area','Year']])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train=X
y_train=y
# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4= nn.Linear(32, 16)
        self.fc5= nn.Linear(16, 8)
        self.fc6 = nn.Linear(8, 1)  # Output layer for regression

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x

# Initialize the model, loss function, and optimizer
model = NeuralNetwork(X_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
epochs = 100
batch_size = 1
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    train_preds = model(X_train_tensor)
    test_preds = model(X_test_tensor)

train_mse = mean_squared_error(y_train_tensor.numpy(), train_preds.numpy())
train_mae = mean_absolute_error(y_train_tensor.numpy(), train_preds.numpy())
test_mse = mean_squared_error(y_test_tensor.numpy(), test_preds.numpy())
test_mae = mean_absolute_error(y_test_tensor.numpy(), test_preds.numpy())

print(f"Training MSE: {train_mse}, Training MAE: {train_mae}")
print(f"Test MSE: {test_mse}, Test MAE: {test_mae}")

# Prepare test data for final predictions
test_data = pd.read_csv('test.csv')

# Feature Engineering for the test dataset
test_data['State'] = le_state.transform(test_data['State'])
test_data['Crop_Type'] = le_crop.transform(test_data['Crop_Type'])
test_data['Soil_Type'] = le_soil.transform(test_data['Soil_Type'])

# Normalize numeric features using the same scaler as the training data
test_data[['Rainfall', 'Irrigation_Area','Year']] = scaler.transform(test_data[['Rainfall', 'Irrigation_Area','Year']])

# Ensure the test data contains the same features as training data
X_test_final = test_data.drop(columns=['id'])

# Convert to tensor and make predictions
X_test_tensor = torch.tensor(X_test_final.values, dtype=torch.float32)
model.eval()
with torch.no_grad():
    test_preds = model(X_test_tensor).numpy()

test_data['Predicted_Crop_Yield'] = test_preds.flatten()

# Save predictions for submission
submission = test_data[['id', 'Predicted_Crop_Yield']].rename(columns={'Predicted_Crop_Yield': 'Target'})
submission.to_csv('submission.csv', index=False)
print("Submission saved to 'submission.csv'")

# Example of calculating MSE and MAE for some actual and predicted values
actual = np.array([4868.0, 5004.0, 4132.0, 4366.0, 4149.0, 975.0, 895.0, 900.0])
predicted = np.array(test_data['Predicted_Crop_Yield'][:8])  # Use the first 8 predictions

mse = np.mean((actual - predicted) ** 2)
mae = np.mean(np.abs(actual - predicted))

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")

# Save the scaler and encoders

joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le_state, 'le_state.pkl')
joblib.dump(le_crop, 'le_crop.pkl')
joblib.dump(le_soil, 'le_soil.pkl')
print("Scaler and preprocessing objects saved.")

# Save the model
torch.save(model.state_dict(), 'neural_network_model.pth')
print("Neural network model saved.")


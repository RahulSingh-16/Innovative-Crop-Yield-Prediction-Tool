from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
import joblib
from torch import nn

app = Flask(__name__)

# Define the neural network class (structure should match the trained model)
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 8)
        self.fc6 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x

# Load pre-trained model and encoders/scalers
input_dim = 6  # The number of features used in training
model = NeuralNetwork(input_dim)
model.load_state_dict(torch.load('neural_network_model.pth'))
model.eval()

scaler = joblib.load('scaler.pkl')
le_state = joblib.load('le_state.pkl')
le_crop = joblib.load('le_crop.pkl')
le_soil = joblib.load('le_soil.pkl')

@app.route('/')
def home():
    # Serve the form
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the POST request
        data = request.json
        print(f"Debug: Received JSON: {data}")  # Debugging log

        # Extract features from the input
        state = le_state.transform([data['state']])[0]
        crop_type = le_crop.transform([data['cropType']])[0]
        soil_type = le_soil.transform([data['soilType']])[0]
        rainfall = float(data['rainfall'])
        irrigation_area = float(data['irrigationArea'])
        year = int(data['year'])

        # Combine the features into a single input array
        inputs = np.array([[year, state, crop_type, rainfall, soil_type, irrigation_area]])

        # Scale numeric features (match indices with those used during training)
        # Select the relevant columns for scaling and reshape them into a 2D array
        inputs[:, [3, 5, 0]] = scaler.transform(inputs[:, [3, 5, 0]])  # Scale inputs[3], inputs[5], inputs[0] together

        # Convert to tensor
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)

        # Make prediction
        with torch.no_grad():
            prediction = model(inputs_tensor).item()

        return jsonify({'yield': round(prediction, 2)})
    except Exception as e:
        print(f"Error: {e}")  # Debugging log for errors
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

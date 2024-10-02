class InverseKinematicsNN(nn.Module):
    def __init__(self):
        super(InverseKinematicsNN, self).__init__()
        self.fc1 = nn.Linear(3, 64)  # Input layer (x, y, z) -> 64 neurons
        self.fc2 = nn.Linear(64, 128)  # Hidden layer -> 128 neurons
        self.fc3 = nn.Linear(128, 256)  # Hidden layer -> 128 neurons
        self.fc4 = nn.Linear(256, 128)  # Hidden layer -> 128 neurons
        self.fc5 = nn.Linear(128, 64)  # Hidden layer -> 64 neurons
        self.fc6 = nn.Linear(64, 3)   # Output layer -> (theta_1, theta_2, theta_3)
        self.leaky_relu = nn.LeakyReLU(0.01)

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        x = self.leaky_relu(self.fc4(x))
        x = self.leaky_relu(self.fc5(x))
        x = self.fc6(x)  # Output angles in radians
        return x



model = InverseKinematicsNN()
model.load_state_dict(torch.load('NN_50_000epoch'))

test_position = torch.tensor([0.2, 0.1, 0.16], dtype=torch.float32)

# Put your model in evaluation mode
model.eval()

# Make prediction
with torch.no_grad():
    predicted_angles = model(test_position)  # Predict the joint angles from the model
    predicted_angles = predicted_angles.detach().numpy()


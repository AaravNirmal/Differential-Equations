import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class TrajectoryMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, dim)
        )

    def forward(self, x):
        return self.net(x)

def train_model(X, Y, dim, epochs=100, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrajectoryMLP(dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X = torch.tensor(X, dtype=torch.float32).to(device)
    Y = torch.tensor(Y, dtype=torch.float32).to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, Y)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

    return model

def predict_trajectory(model, x0, steps):
    model.eval()
    trajectory = [x0]
    x = torch.tensor(x0, dtype=torch.float32).unsqueeze(0)
    for _ in range(steps):
        with torch.no_grad():
            x = model(x)
        trajectory.append(x.cpu().numpy().flatten())
    return np.array(trajectory)

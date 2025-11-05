import os
import torch
import numpy as np
from model import SimpleMLP

DIM = 4  # set your dimension here
MODEL_PATH = "results/ml_model.pt"

model = SimpleMLP(dim=DIM)

# 🔁 Load existing model if it exists
if os.path.exists(MODEL_PATH):
    print("Loading previous model...")
    model.load_state_dict(torch.load(MODEL_PATH))

# Load and stack all datasets
trajectories = []
for i in range(30):
    path = f"results/real_{i}.npy"
    if os.path.exists(path):
        X = np.load(path)
        X_train = X[:-1]
        Y_train = X[1:]
        trajectories.append((X_train, Y_train))

# Combine all into one dataset
X_all = np.concatenate([x for x, y in trajectories], axis=0)
Y_all = np.concatenate([y for x, y in trajectories], axis=0)

# Convert to tensors
X_all = torch.tensor(X_all, dtype=torch.float32)
Y_all = torch.tensor(Y_all, dtype=torch.float32)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(X_all)
    loss = loss_fn(out, Y_all)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# ✅ Save the updated model
torch.save(model.state_dict(), MODEL_PATH)

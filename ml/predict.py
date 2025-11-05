import torch
import numpy as np
from model import SimpleMLP

X0 = np.load("results/real.npy")[0]  # starting point
model = SimpleMLP(dim=len(X0))
model.load_state_dict(torch.load("results/ml_model.pt"))
model.eval()

trajectory = [X0]
x = torch.tensor(X0, dtype=torch.float32).unsqueeze(0)

for _ in range(500):
    with torch.no_grad():
        x = model(x)
    trajectory.append(x.squeeze().numpy())

trajectory = np.array(trajectory)
np.save("results/predicted.npy", trajectory)

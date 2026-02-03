import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from src.data.dataset import generate_synthetic_anomaly_data
from src.models.vae import VAE
from src.training.train import train
from src.evaluation.metrics import evaluate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LATENT_DIMS = [2, 4, 8, 16]
EPOCHS = 50
BATCH_SIZE = 128
LR = 1e-3

X_train, X_test, y_test = generate_synthetic_anomaly_data()

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
    batch_size=BATCH_SIZE,
    shuffle=True
)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

results = []

for ld in LATENT_DIMS:
    model = VAE(input_dim=30, latent_dim=ld).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train(model, train_loader, optimizer, EPOCHS, DEVICE)

    with torch.no_grad():
        recon, _, _ = model(X_test_tensor)
        scores = torch.mean((X_test_tensor - recon) ** 2, dim=1).cpu().numpy()

    metrics = evaluate(y_test, scores)
    metrics["Latent_Dim"] = ld
    results.append(metrics)

df = pd.DataFrame(results)
os.makedirs("experiments", exist_ok=True)
df.to_csv("experiments/results.csv", index=False)

print(df)

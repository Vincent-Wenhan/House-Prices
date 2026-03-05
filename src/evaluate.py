import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.preprocess import load_data, preprocess_data
import pandas as pd
import numpy as np

def evaluate(cfg, model, data_loader, device, use_log1p=True, labels=None):
    model.eval()
    total_loss = 0
    y_preds = []
    loss_fn = getattr(nn, cfg.train.loss)()
    with torch.no_grad():
        if labels is not None:
            for X_batch, y_batch in data_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                total_loss += loss.item() * X_batch.size(0)
                if use_log1p:
                    y_pred = torch.expm1(y_pred)
                y_preds.append(y_pred.cpu())
            y_preds = torch.cat(y_preds, dim=0)
            rmse = ((total_loss / len(data_loader.dataset)) ** 0.5)
            return y_preds, rmse

def predict(cfg, model, device, use_log1p=True, out_file="submission.csv"):
    model.eval()
    train_data, test_data = load_data(cfg)
    train_features, train_labels, test_features = preprocess_data(cfg, train_data, test_data)
    test_dataset = TensorDataset(test_features)
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False)

    y_preds = []
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch[0].to(device)
            y_pred = model(X_batch)
            if use_log1p:
                y_pred = torch.expm1(y_pred)
            y_preds.append(y_pred.cpu())
        y_preds = torch.cat(y_preds, dim=0).numpy().flatten()

    submission = pd.DataFrame(
        {"Id": test_data.Id, "SalePrice": y_preds}
    )
    
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    submission.to_csv(out_file, index=False)
    print(f"Submission file saved to {out_file}")
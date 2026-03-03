import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import hydra
from omegaconf import DictConfig
from src.preprocess import load_data, preprocess_data
from src.model import MLP

@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    # load and preprocess the data
    train_data, test_data = load_data(cfg)
    train_features, train_labels, test_features = preprocess_data(cfg, train_data, test_data)

    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)

    # initialize the model, loss function, and optimizer
    model = MLP(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = getattr(torch.optim, cfg.train.optimizer)(model.parameters(), lr=cfg.train.lr)
    loss_fn = getattr(nn, cfg.train.loss)()

    for epoch in range(cfg.train.num_epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
        epoch_loss /= len(train_loader.dataset)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{cfg.train.num_epochs}, Loss: {epoch_loss:.4f}")



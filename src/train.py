import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import hydra
from omegaconf import DictConfig
from src.preprocess import load_data, preprocess_data
from src.model import MLP
from src.evaluate import evaluate
from sklearn.model_selection import KFold

def train(cfg: DictConfig, device: torch.device):
    # load and preprocess the data
    train_data, test_data = load_data(cfg)
    train_features, train_labels, test_features = preprocess_data(cfg, train_data, test_data)

    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)

    # initialize the model, loss function, and optimizer
    model = MLP(cfg)
    model = model.to(device)
    optimizer = getattr(torch.optim, cfg.train.optimizer)(model.parameters(), lr=cfg.train.learning_rate)
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
    return model

def k_fold_cv(cfg: DictConfig, device: torch.device):
    "k-fold cross validation"
    train_data, test_data = load_data(cfg)
    train_features, train_labels, test_features = preprocess_data(cfg, train_data, test_data)

    kfold = KFold(n_splits=cfg.train.k_folds, shuffle=True, random_state=42)
    fold_rmse = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_features)):
        print(f"Fold {fold+1}/{cfg.train.k_folds}")
        X_train, y_train = train_features[train_idx], train_labels[train_idx]
        X_val, y_val = train_features[val_idx], train_labels[val_idx]
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=cfg.train.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=cfg.train.batch_size, shuffle=False)
        model = MLP(cfg)
        model = model.to(device)
        optimizer = getattr(torch.optim, cfg.train.optimizer)(model.parameters(), lr=cfg.train.learning_rate)
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

        y_preds, rmse = evaluate(cfg, model, val_loader, device, use_log1p=cfg.preprocess.use_log1p, labels=y_val)
        fold_rmse.append(rmse)
        print(f"Fold {fold+1} RMSE: {rmse:.4f}")
    
    avg_rmse = sum(fold_rmse) / cfg.train.k_folds
    print(f"Average RMSE over {cfg.train.k_folds} folds: {avg_rmse:.4f}")
    return avg_rmse


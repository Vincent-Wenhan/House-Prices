import torch
import torch.nn as nn

def evaluate(cfg, model, data_loader, device, use_log1p=False, labels=None):
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


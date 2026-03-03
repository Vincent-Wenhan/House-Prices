import os
import torch
import pandas as pd
import hydra
from omegaconf import DictConfig

def load_data(cfg: DictConfig):
    "Load the train and test data from the specified paths in the configuration."
    train_path = os.path.join(cfg.data.dir, cfg.data.train_file)
    test_path = os.path.join(cfg.data.dir, cfg.data.test_file)
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

def preprocess_data(cfg: DictConfig, train_data: pd.DataFrame, test_data: pd.DataFrame):
    "Preprocess the train and test data by standardizing numeric features and one-hot encoding categorical features."
    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

    # standardize the numeric features
    numeric_features = all_features.dtypes[all_features.dtypes != "object"].index
    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda x: (x - x.mean()) / x.std()
    )
    all_features[numeric_features] = all_features[numeric_features].fillna(0)

    # one-hot encode the categorical features
    all_features = pd.get_dummies(all_features, dummy_na=True)

    # split the data back into train and test sets and convert to PyTorch tensors
    n_tain = train_data.shape[0]
    train_features = torch.tensor(all_features[:n_tain].values, dtype=torch.float32)
    test_features = torch.tensor(all_features[n_tain:].values, dtype=torch.float32)
    train_labels = torch.tensor(train_data.SalePrice.values, dtype=torch.float32).view(-1, 1)
    return train_features, train_labels, test_features

import os
import hydra
import torch
from omegaconf import DictConfig
from src.train import train, k_fold_cv
from src.evaluate import predict

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = os.path.join(cfg.outdir, "submission.csv")
    if cfg.train.is_kfold and cfg.train.k_folds > 0:
        print(f"Running {cfg.train.k_folds}-fold cross-validation...")
        avg_rmse = k_fold_cv(cfg, device)
    
    if cfg.train.is_train:
        print("Training the final model...")
        model = train(cfg, device)

        if cfg.train.is_predict:
            print("Predicting on test set...")
            predict(cfg, model, device, use_log1p=cfg.preprocess.use_log1p, out_file=out_path)

if __name__ == "__main__":
    main()
    
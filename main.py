import hydra
from omegaconf import DictConfig
from src.train import train, evaluate, k_fold_cv

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    if cfg.train.is_kfold and cfg.train.k_folds > 0:
        print(f"Running {cfg.train.k_folds}-fold cross-validation...")
        avg_rmse = k_fold_cv(cfg)
    
    if cfg.train.is_train:
        print("Training the final model...")
        train(cfg)

if __name__ == "__main__":
    main()
    
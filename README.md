# 🏠 House Prices Prediction

This project implements the Kaggle competition:
👉 [https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

---

# ⚙️ Initializtion

Install dependencies:

```bash
pip install requirements.txt
# pip install torch pandas scikit-learn hydra-core
```

---

# 🚀 How to Run

All commands should be executed in the project root directory.

---

## 1️⃣ Run K-Fold Cross Validation

```bash
python main.py training.is_kfold=true training.k_folds=5 training.is_train=false
```

This will:

1. Load and preprocess data
2. Perform K-Fold cross validation 
---


## 2️⃣ Run Only Final Training and Predict Test Set

```bash
python main.py training.is_kfold=false training.is_train=true
```

Use this when generating final submission.

---

# 🔧 Hydra Configuration

You can override parameters from command line:

### Change learning rate

```bash
python main.py training.lr=0.0005
```

### Change hidden layers

```bash
python main.py model.hidden_sizes=[256,128]
```

### Change dropout

```bash
python main.py model.dropout=0.3
```

---

# 📈 Evaluation Metric

The competition metric is:

```
RMSE of log(SalePrice)
```

If `use_log1p=true`, labels are transformed using:

```
log1p(SalePrice)
```

And predictions are transformed back using:

```
expm1()
```

---

# 🏆 Competition Link

**Kaggle Competition:**

👉 [https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

Full name:

**House Prices – Advanced Regression Techniques**

This competition focuses on predicting final house sale prices using 79 explanatory variables describing residential homes.

---

# 📂 Dataset Link

Competition Data Page:

👉 [https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)


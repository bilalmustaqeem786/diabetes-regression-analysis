# Diabetes Regression Analysis

A regression project that predicts diabetes disease progression one year after baseline from ten clinical measurements, comparing Linear Regression and Ridge regression with correlation analysis and residual diagnostics.

> **Note:** This project is for learning and portfolio demonstration only. It is **not** clinical or medical advice.

## Features

- Exploratory analysis of the diabetes dataset (442 samples, 10 features)
- Correlation analysis against the progression target (strongest: bmi 0.586, s5 0.566)
- Train/test split (80/20) with Linear Regression and Ridge (alpha = 1.0) compared side by side
- MAE, RMSE and R² evaluation for both models
- Residual diagnostics for the Ridge model with a saved residual plot

## Tech Stack

- Python
- pandas, NumPy, scikit-learn, matplotlib

## Results

Test set: 89 samples (20% of 442).

| Model | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Linear Regression | 42.79 | 53.85 | **0.4526** |
| Ridge (alpha = 1.0) | 46.14 | 55.47 | 0.4192 |

Ridge residuals: mean −0.54, standard deviation 55.79.

## Project Structure

```
diabetes-regression-analysis/
├── src/
│   └── train.py          # Trains both models, prints metrics, saves the residual plot
├── notebooks/
│   └── 02_diabetes_progression_regression.ipynb
├── outputs/
│   └── residual_plot.png
├── requirements.txt
├── LICENSE
└── README.md
```

## How to Run

```bash
# 1. Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the training script
python src/train.py
```

The script prints both models' metrics and saves the Ridge residual plot to `outputs/residual_plot.png`.

## Dataset

The scikit-learn built-in diabetes dataset (`sklearn.datasets.load_diabetes`) — no download and no real patient records are used. The target is a quantitative measure of disease progression one year after baseline.

## Author

**Bilal Ahmad — Healthcare Data Analyst**

- [LinkedIn](https://www.linkedin.com/in/bilal-ahmad-7ab444250)
- [Kaggle](https://www.kaggle.com/bilalmustaqeem)

## License

MIT — see [LICENSE](LICENSE).

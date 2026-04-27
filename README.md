# 🏠 House Price Prediction using Linear Regression

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square)

---

## 📋 Project Overview

A complete end-to-end machine learning project that predicts house prices using **Linear Regression**. The project covers the full data science pipeline, from raw data to model evaluation, with detailed visualizations at every step.

**Business Question:** *What factors most influence house prices, and how accurately can we predict them?*

---

## 🎯 Key Highlights

| Metric | Result |
|--------|--------|
| **Algorithm** | Linear Regression |
| **Dataset** | California Housing Dataset (20,640 records) |
| **R² Score** | ~0.60 (60% variance explained) |
| **Features Used** | 8 (Income, Age, Rooms, Population, Location, etc.) |
| **Train/Test Split** | 80% / 20% |

---

## 🛠️ Tools & Libraries Used

| Tool | Purpose |
|------|---------|
| **Python** | Core programming language |
| **Pandas** | Data manipulation and cleaning |
| **NumPy** | Numerical computations |
| **Matplotlib** | Data visualization |
| **Seaborn** | Statistical visualizations |
| **Scikit-Learn** | Machine learning model |

---

## 📊 Project Steps

```
1. Data Loading           → Loaded California Housing Dataset (20,640 rows × 9 cols)
2. Exploratory Data Analysis (EDA)
   ├── Distribution plots
   ├── Correlation heatmap
   ├── Feature vs target scatter plots
   └── Missing value check
3. Data Preprocessing
   ├── Feature/Target split (X and y)
   ├── Train-Test Split (80/20)
   └── StandardScaler normalization
4. Model Building         → LinearRegression() from Scikit-Learn
5. Model Evaluation       → R² Score, MSE, RMSE
6. Visualization          → Actual vs Predicted, Residuals Plot
7. Business Insights      → Key features driving house prices
```

---

## 🔍 Key Findings

- **Median Income** is the strongest predictor of house price (highest positive correlation)
- **Location** (Latitude/Longitude) significantly impacts prices, coastal areas are more expensive
- **House Age** has a surprisingly weak correlation with price
- **Average Rooms** per household positively correlates with price but with diminishing returns
- The model achieves **~60% accuracy**, a reasonable baseline for Linear Regression

---

## 📁 Files in This Repository

```
house-price-prediction/
│
├── house_price_prediction.py    ← Main Python script (run this!)
├── eda_visualizations.png       ← EDA charts (auto-generated)
├── model_results.png            ← Model performance charts (auto-generated)
└── README.md                    ← This file
```

---

## ▶️ How to Run This Project

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run the script
```bash
python house_price_prediction.py
```

The script will:
1. Automatically download the California Housing dataset (no manual download needed!)
2. Perform complete EDA with visualizations
3. Train the Linear Regression model
4. Evaluate and display results
5. Save two PNG charts to your folder

---

## 📈 Sample Output

```
✅ Dataset loaded! Shape: (20640, 9)
   → 20640 rows (houses) and 9 columns (features)

📈 Model Performance Metrics:
   R² Score (Accuracy) : 0.5758  → Model explains 57.6% of price variance
   MSE                 : 0.5559
   RMSE                : 0.7456  → Average error of $74,560

✅ Model Performance: GOOD, R² above 0.6
```

---

## 💡 Future Improvements

- [ ] Try Random Forest & XGBoost for better accuracy
- [ ] Add feature engineering (price per room, income-to-price ratio)
- [ ] Build an interactive web app using Streamlit
- [ ] Hyperparameter tuning with GridSearchCV

---

## 👩‍💻 Author

**Gursheen Kaur** | Data Analyst | MBA Business Analytics

📧 gursheensran@gmail.com | 📍 Patiala, Punjab, India

---

*⭐ If you found this project helpful, please give it a star!*

# =============================================================
# House Price Prediction using Linear Regression
# Author: Gursheen Kaur
# Tools: Python, Pandas, Scikit-Learn, Matplotlib, Seaborn
# =============================================================

# -----------------------------------------------
# STEP 1: Import Libraries
# -----------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing  # Free dataset, no download needed
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

print("✅ All libraries imported successfully!")
print("=" * 60)

# -----------------------------------------------
# STEP 2: Load Dataset
# -----------------------------------------------
print("\n📦 Loading California Housing Dataset...")

housing = fetch_california_housing(as_frame=True)
df = housing.frame

print(f"✅ Dataset loaded! Shape: {df.shape}")
print(f"   → {df.shape[0]} rows (houses) and {df.shape[1]} columns (features)")

# -----------------------------------------------
# STEP 3: Exploratory Data Analysis (EDA)
# -----------------------------------------------
print("\n" + "=" * 60)
print("📊 EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 60)

# 3.1 Basic Info
print("\n🔍 First 5 rows of data:")
print(df.head())

print("\n📋 Dataset Info:")
print(df.info())

print("\n📈 Statistical Summary:")
print(df.describe().round(2))

# 3.2 Check for Missing Values
print("\n🔎 Checking for Missing Values:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("✅ No missing values found! Data is clean.")
else:
    print(missing[missing > 0])

# 3.3 Correlation Analysis
print("\n🔗 Correlation with House Price (MedHouseVal):")
correlation = df.corr()["MedHouseVal"].sort_values(ascending=False)
print(correlation.round(3))

# -----------------------------------------------
# STEP 4: Data Visualization
# -----------------------------------------------
print("\n" + "=" * 60)
print("📊 CREATING VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("House Price Prediction - Exploratory Data Analysis\nby Gursheen Kaur",
             fontsize=14, fontweight='bold', y=1.02)

# Plot 1: Distribution of House Prices
axes[0, 0].hist(df["MedHouseVal"], bins=50, color="#4C72B0", edgecolor="white", alpha=0.8)
axes[0, 0].set_title("Distribution of House Prices", fontweight='bold')
axes[0, 0].set_xlabel("Median House Value ($100k)")
axes[0, 0].set_ylabel("Frequency")
axes[0, 0].axvline(df["MedHouseVal"].mean(), color='red', linestyle='--',
                    label=f'Mean: {df["MedHouseVal"].mean():.2f}')
axes[0, 0].legend()

# Plot 2: Income vs House Price
axes[0, 1].scatter(df["MedInc"], df["MedHouseVal"], alpha=0.3, color="#DD8452", s=10)
axes[0, 1].set_title("Income vs House Price", fontweight='bold')
axes[0, 1].set_xlabel("Median Income")
axes[0, 1].set_ylabel("Median House Value")

# Plot 3: Correlation Heatmap
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            ax=axes[0, 2], center=0, square=True, cbar_kws={"shrink": 0.8})
axes[0, 2].set_title("Feature Correlation Heatmap", fontweight='bold')

# Plot 4: House Age Distribution
axes[1, 0].hist(df["HouseAge"], bins=30, color="#55A868", edgecolor="white", alpha=0.8)
axes[1, 0].set_title("Distribution of House Age", fontweight='bold')
axes[1, 0].set_xlabel("House Age (years)")
axes[1, 0].set_ylabel("Frequency")

# Plot 5: Rooms per Household vs Price
axes[1, 1].scatter(df["AveRooms"], df["MedHouseVal"], alpha=0.3, color="#C44E52", s=10)
axes[1, 1].set_xlim(0, 20)
axes[1, 1].set_title("Average Rooms vs House Price", fontweight='bold')
axes[1, 1].set_xlabel("Average Rooms per Household")
axes[1, 1].set_ylabel("Median House Value")

# Plot 6: Feature Importance (Correlation with Price)
feature_corr = abs(df.corr()["MedHouseVal"]).drop("MedHouseVal").sort_values()
colors = ["#4C72B0" if c > 0 else "#C44E52" for c in
          df.corr()["MedHouseVal"].drop("MedHouseVal")[feature_corr.index]]
axes[1, 2].barh(feature_corr.index, feature_corr.values, color=colors, edgecolor="white")
axes[1, 2].set_title("Feature Importance\n(Correlation with Price)", fontweight='bold')
axes[1, 2].set_xlabel("|Correlation Coefficient|")

plt.tight_layout()
plt.savefig("eda_visualizations.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Visualization saved as 'eda_visualizations.png'")

# -----------------------------------------------
# STEP 5: Data Preprocessing
# -----------------------------------------------
print("\n" + "=" * 60)
print("⚙️  DATA PREPROCESSING")
print("=" * 60)

# Define features (X) and target (y)
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

print(f"✅ Features (X): {list(X.columns)}")
print(f"✅ Target (y): MedHouseVal (Median House Value in $100k)")

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n✅ Data Split:")
print(f"   Training set  : {X_train.shape[0]} samples (80%)")
print(f"   Testing set   : {X_test.shape[0]} samples (20%)")

# Scale features for better performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Features scaled using StandardScaler")

# -----------------------------------------------
# STEP 6: Build & Train Linear Regression Model
# -----------------------------------------------
print("\n" + "=" * 60)
print("🤖 BUILDING LINEAR REGRESSION MODEL")
print("=" * 60)

model = LinearRegression()
model.fit(X_train_scaled, y_train)
print("✅ Model trained successfully!")

# -----------------------------------------------
# STEP 7: Model Evaluation
# -----------------------------------------------
print("\n" + "=" * 60)
print("📊 MODEL EVALUATION")
print("=" * 60)

y_pred = model.predict(X_test_scaled)

mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print(f"\n📈 Model Performance Metrics:")
print(f"   R² Score (Accuracy) : {r2:.4f}  → Model explains {r2*100:.1f}% of price variance")
print(f"   MSE                 : {mse:.4f}")
print(f"   RMSE                : {rmse:.4f}  → Average error of ${rmse*100000:.0f}")

if r2 > 0.6:
    print("\n✅ Model Performance: GOOD — R² above 0.6")
elif r2 > 0.4:
    print("\n⚠️  Model Performance: MODERATE — Consider feature engineering")
else:
    print("\n❌ Model Performance: POOR — Try other algorithms")

# Feature Coefficients
print(f"\n🔍 Feature Coefficients (Impact on House Price):")
coef_df = pd.DataFrame({
    "Feature"    : X.columns,
    "Coefficient": model.coef_
}).sort_values("Coefficient", ascending=False)

for _, row in coef_df.iterrows():
    direction = "↑ Increases" if row["Coefficient"] > 0 else "↓ Decreases"
    print(f"   {row['Feature']:20s}: {row['Coefficient']:+.4f}  ({direction} price)")

# -----------------------------------------------
# STEP 8: Actual vs Predicted Plot
# -----------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Linear Regression Model Results\nby Gursheen Kaur",
             fontsize=13, fontweight='bold')

# Plot 1: Actual vs Predicted
axes[0].scatter(y_test, y_pred, alpha=0.4, color="#4C72B0", s=15)
axes[0].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             'r--', lw=2, label="Perfect Prediction")
axes[0].set_xlabel("Actual House Price ($100k)")
axes[0].set_ylabel("Predicted House Price ($100k)")
axes[0].set_title(f"Actual vs Predicted\nR² = {r2:.4f}", fontweight='bold')
axes[0].legend()

# Plot 2: Residuals Plot
residuals = y_test - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.4, color="#DD8452", s=15)
axes[1].axhline(y=0, color='red', linestyle='--', lw=2)
axes[1].set_xlabel("Predicted Values")
axes[1].set_ylabel("Residuals (Actual - Predicted)")
axes[1].set_title("Residuals Plot\n(Should be randomly scattered around 0)", fontweight='bold')

plt.tight_layout()
plt.savefig("model_results.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Model results saved as 'model_results.png'")

# -----------------------------------------------
# STEP 9: Sample Predictions
# -----------------------------------------------
print("\n" + "=" * 60)
print("🏠 SAMPLE PREDICTIONS")
print("=" * 60)

sample = X_test.head(5)
sample_scaled = scaler.transform(sample)
predictions = model.predict(sample_scaled)
actual = y_test.head(5).values

print("\n  House | Predicted Price | Actual Price | Difference")
print("  " + "-" * 55)
for i, (pred, act) in enumerate(zip(predictions, actual)):
    diff = pred - act
    sign = "+" if diff > 0 else ""
    print(f"  {i+1:5d} | ${pred*100000:>12,.0f} | ${act*100000:>11,.0f} | {sign}${diff*100000:>+10,.0f}")

print("\n" + "=" * 60)
print("✅ Analysis Complete!")
print("=" * 60)
print("\n📁 Files Generated:")
print("   → eda_visualizations.png  (EDA charts)")
print("   → model_results.png       (Model performance charts)")
print("\n👩‍💻 Author: Gursheen Kaur | Data Analyst")
print("📧 gursheensran@gmail.com")

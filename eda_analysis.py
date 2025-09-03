import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import kagglehub
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------- Download dataset -----------------
path = kagglehub.dataset_download("surajjha101/cuisine-rating")
print("Dataset path:", path)

# List files to detect correct CSV
files = os.listdir(path)
print("Files inside dataset folder:", files)

# Pick first CSV file
csv_file = [f for f in files if f.endswith(".csv")][0]
csv_path = os.path.join(path, csv_file)

# ----------------- Load dataset -----------------
df = pd.read_csv(csv_path, encoding="latin-1")
print("\nData Loaded Successfully!\n")

# ----------------- Basic EDA -----------------
print(df.info())
print(df.describe())

# Missing values
print("\nMissing values:\n", df.isnull().sum())

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("correlation.png")
plt.show()

# Histograms
df.hist(figsize=(12, 10))
plt.tight_layout()
plt.savefig("histograms.png")
plt.show()

# Pairplot (sampled for speed)
sns.pairplot(df.sample(min(200, len(df))))
plt.savefig("pairplot.png")
plt.show()

# ----------------- Predictive Modeling -----------------
# Drop User ID (not predictive if present)
if "User ID" in df.columns:
    df = df.drop(columns=["User ID"])

# Features & target
X = df.drop(columns=["Overall Rating"])
y = df["Overall Rating"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----- Linear Regression -----
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# ----- Random Forest -----
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# ----------------- Evaluation -----------------
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n{model_name} Performance:")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²: {r2:.3f}")

evaluate_model(y_test, y_pred_lr, "Linear Regression")
evaluate_model(y_test, y_pred_rf, "Random Forest")

# ----------------- Feature Importance -----------------
feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
feat_importances.sort_values().plot(
    kind="barh", figsize=(8, 5), title="Feature Importance (Random Forest)"
)
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("clean_laptops.csv")
df.columns = df.columns.str.lower()

df['ram_gb'] = pd.to_numeric(df['ram_gb'], errors='coerce')
df['price'] = pd.to_numeric(df['price'], errors='coerce')

df['ram_gb'] = df['ram_gb'].fillna(df['ram_gb'].median())

plt.figure(figsize=(8,5))
sns.histplot(df['price'], kde=True)
plt.title("Price Distribution of Laptops")
plt.xlabel("Price")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(x=df['ram_gb'], y=df['price'])
plt.title("RAM(GB)")
plt.ylabel("Price")
plt.show()

plt.figure(figsize=(10,6))
sns.barplot(x=df['brand'], y=df['price'])
plt.xticks(rotation=45)
plt.title("Brand vs Laptop Price")
plt.xlabel("Brand")
plt.ylabel("Average Price")
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

y = df['price']
X = df.drop('price', axis=1)

X = pd.get_dummies(X, drop_first=True)

print(df.info())
print(df.head())
print(df.isnull().sum())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(y_pred[:5])

print("--- Random Forest ---")
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linestyle='--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Random Forest: Actual vs Predicted")
plt.show()

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print(" ---Linear Regression---- ")
print("R2 Score:", r2_score(y_test, y_pred_lr))
print("MAE:", mean_absolute_error(y_test, y_pred_lr))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))

plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred_lr, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linestyle='--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear Regression: Actual vs Predicted")
plt.show()

importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='importance', y='feature', data=importance)
plt.title("Feature Importance")
plt.show()

print("\n----- MODEL COMPARISON ----")
print("Random Forest R2:", round(r2_score(y_test, y_pred), 3))
print("Linear Regression R2:", round(r2_score(y_test, y_pred_lr), 3))
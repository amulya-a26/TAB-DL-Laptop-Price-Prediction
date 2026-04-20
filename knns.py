import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("laptop_price.csv")
df['ram_gb'] = pd.to_numeric(df['ram_gb'], errors='coerce')
df['ram_gb'] = df['ram_gb'].fillna(df['ram_gb'].median())
df = pd.get_dummies(df, drop_first=True)
X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
# 1. Actual vs Predicted
plt.figure()
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linestyle='--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("KNN: Actual vs Predicted")
plt.show()
# 2. Residual Plot
residuals = y_test - y_pred
plt.figure()
plt.scatter(y_pred, residuals)
plt.axhline(y=0)
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residual Plot (KNN)")
plt.show()
# 3. K vs R2 Score graph
scores = []
k_values = range(1, 15)
for k in k_values:
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    scores.append(r2_score(y_test, pred))
plt.figure()
plt.plot(k_values, scores, marker='o')
plt.xlabel("K value")
plt.ylabel("R2 Score")
plt.title("K vs R2 Score")
plt.show()
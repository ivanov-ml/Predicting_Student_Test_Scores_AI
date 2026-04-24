from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
import joblib

data = pd.read_csv('train_data_cleaned_scaled.csv')
X = data.drop(['exam_score'], axis=1)
y = data['exam_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# alpha - это коэффициент регуляризации (чем выше, тем сильнее штраф)
model = Lasso(alpha=0.01)
model.fit(X, y)


y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f'MAE на обучающей выборке: {train_mae:.2f}')
print(f'MAE на тестовой выборке: {test_mae:.2f}')
print(f'MSE на обучающей выборке: {train_mse:.2f}')
print(f'MSE на тестовой выборке: {test_mse:.2f}')
print(f'RMSE на обучающей выборке: {train_rmse:.2f}')
print(f'RMSE на тестовой выборке: {test_rmse:.2f}')
print(f'R² на обучающей выборке: {train_r2:.4f}')
print(f'R² на тестовой выборке: {test_r2:.4f}\n\n')
print(f'R² on the test sample: {test_r2:.4f}, MAE on the test sample: {test_mae:.2f}')
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
print(f"CV R²: {scores.mean():.4f} (+/- {scores.std():.4f})")
model_name = 'LassoRegression.pkl'
joblib.dump(model, model_name)
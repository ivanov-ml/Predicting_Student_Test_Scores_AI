import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import joblib
import mlflow
import mlflow.sklearn

# 1. Загрузка данных
data = pd.read_csv('data/playground-series-s6e1/train_data_cleaned_scaled.csv')
X = data.drop(['exam_score'], axis=1)
y = data['exam_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Параметры модели
params = {
    'n_estimators': 2000,
    'learning_rate': 0.05,
    'random_state': 42
}
mlflow.set_tracking_uri("http://127.0.0.1:5000")
# 3. Обучение с логированием в MLflow
with mlflow.start_run():
    # Логируем параметры
    mlflow.log_params(params)

    # Обучаем модель
    model = lgb.LGBMRegressor(**params, verbose=0)
    model.fit(X_train, y_train)

    # Предсказания
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Метрики
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # Логируем метрики
    mlflow.log_metric("train_r2", train_r2)
    mlflow.log_metric("test_r2", test_r2)
    mlflow.log_metric("test_mae", test_mae)
    mlflow.log_metric("test_rmse", test_rmse)

    # Логируем важность признаков (график)
    importance = model.feature_importances_
    feature_names = X.columns
    imp_df = pd.DataFrame({'feature': feature_names, 'importance': importance}).sort_values('importance',
                                                                                            ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(imp_df['feature'].head(10), imp_df['importance'].head(10))
    plt.xlabel('Importance')
    plt.title('Top 10 Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

    # Сохраняем график как артефакт
    mlflow.log_artifact('feature_importance.png')

    # Сохраняем модель (в формате MLflow)
    mlflow.sklearn.log_model(model, "lgbm_model")

    # Сохраняем модель через joblib (как ты делал)
    joblib.dump(model, 'LGBMRegressor.pkl')
    mlflow.log_artifact('LGBMRegressor.pkl')

    # Вывод метрик в консоль
    print(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}, MAE: {test_mae:.2f}")
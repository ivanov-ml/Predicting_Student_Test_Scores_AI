import joblib
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import joblib


# ========== 1. ЗАГРУЗКА СОХРАНЕННЫХ МОДЕЛЕЙ ==========
models = {
    'hgbr': joblib.load('HistGradientBoostingRegressor.pkl'),
    'xgb': joblib.load('XGBRegressor.pkl'),
    'lgbm': joblib.load('LGBMRegressor.pkl'),
}

print("✅ Модели загружены:")
for name, model in models.items():
    print(f"  - {name}: {type(model).__name__}")


data = pd.read_csv('train_data_cleaned_scaled.csv')
X = data.drop(['exam_score'], axis=1)
y = data['exam_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


voting_ensemble = VotingRegressor([
    ('xgb', models['xgb']),
    ('lgbm', models['lgbm']),
    ('hgbr', models['hgbr'])
], weights=[2, 2, 1])  # XGB и LGBM важнее
voting_ensemble.fit(X_train, y_train)  # модели переобучатся быстро (просто подгрузят веса)

y_pred_voting = voting_ensemble.predict(X_test)

print(f"  R²: {r2_score(y_test, y_pred_voting):.4f}")
print(f"  MAE: {mean_absolute_error(y_test, y_pred_voting):.2f}")


joblib.dump(voting_ensemble, 'voting_ensemble.pkl')
print("✅ Ансамбль сохранён в 'voting_ensemble.pkl'")
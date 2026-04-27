import joblib
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import joblib


# ========== 1. ЗАГРУЗКА СОХРАНЕННЫХ МОДЕЛЕЙ ==========
models = {
    'hgbr': joblib.load('models/HistGradientBoostingRegressor.pkl'),
    'xgb': joblib.load('models/XGBRegressor.pkl'),
    'lgbm': joblib.load('models/LGBMRegressor.pkl'),
}

print("✅ Модели загружены:")
for name, model in models.items():
    print(f"  - {name}: {type(model).__name__}")


data = pd.read_csv('data/playground-series-s6e1/train_data_cleaned_scaled.csv')
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

test_r2 = r2_score(y_test, y_pred_voting)
test_mae = mean_absolute_error(y_test, y_pred_voting)
print(f'R² on the test sample: {test_r2:.4f}, MAE on the test sample: {test_mae:.2f}')
#scores = cross_val_score(voting_ensemble, X_train, y_train, cv=5, scoring='r2')
#print(f"CV R²: {scores.mean():.4f} (+/- {scores.std():.4f})")


joblib.dump(voting_ensemble, 'models/voting_ensemble.pkl')
print("✅ Ансамбль сохранён в 'voting_ensemble.pkl'")
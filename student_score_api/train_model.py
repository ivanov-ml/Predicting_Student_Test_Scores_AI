import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb

print("Загрузка данных...")
data = pd.read_csv('/Users/dmitrii/PycharmProjects/Students_score/data/playground-series-s6e1/train.csv')

# Кодирование категориальных признаков
data['gender'] = data['gender'].map({'female': 0, 'male': 1, 'other': 2})
data['study_method'] = data['study_method'].map({'online videos': 0, 'self-study': 1, 'coaching': 2, 'group study': 3, 'mixed': 4})
data['exam_difficulty'] = data['exam_difficulty'].map({'easy': 0, 'moderate': 1, 'hard': 2})
data['facility_rating'] = data['facility_rating'].map({'low': 0, 'medium': 1, 'high': 2})
data['sleep_quality'] = data['sleep_quality'].map({'poor': 0, 'average': 1, 'good': 2})
data['internet_access'] = data['internet_access'].map({'no': 0, 'yes': 1})
data['course'] = data['course'].map({'b.sc': 0, 'diploma': 1, 'bca': 2, 'b.com': 3, 'ba': 4, 'bba': 5, 'b.tech': 6})

# Удаляем ненужные колонки
columns_to_drop = ['id', 'age', 'gender', 'course', 'internet_access', 'exam_difficulty']
X = data.drop(columns=columns_to_drop + ['exam_score'], axis=1)
y = data['exam_score']

# Признаки для масштабирования
features_to_scale = ['study_hours', 'class_attendance', 'sleep_hours',
                     'sleep_quality', 'study_method', 'facility_rating']

# Масштабирование
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[features_to_scale] = scaler.fit_transform(X[features_to_scale])

# Обучение модели
print("Обучение модели...")
model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Сохранение
joblib.dump(model, 'models/LGBMRegressor.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print("✅ Модель и scaler сохранены в папку models/")
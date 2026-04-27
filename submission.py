import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# ========== 1. ЗАГРУЗКА МОДЕЛИ ==========
print("Загрузка модели...")
voting_ensemble = joblib.load('models/LGBMRegressor.pkl')
print("✅ Модель загружена")

# ========== 2. ЗАГРУЗКА ТЕСТОВЫХ ДАННЫХ ==========
row_data = pd.read_csv('data/playground-series-s6e1/test.csv')
passenger_ids = row_data['id'].copy()

# ========== 3. ПРЕДОБРАБОТКА ==========
row_data['gender'] = row_data['gender'].map({'female': 0, 'male': 1, 'other': 2})
row_data['study_method'] = row_data['study_method'].map({'online videos': 0, 'self-study': 1, 'coaching': 2, 'group study': 3, 'mixed': 4})
row_data['exam_difficulty'] = row_data['exam_difficulty'].map({'easy': 0, 'moderate': 1, 'hard': 2})
row_data['facility_rating'] = row_data['facility_rating'].map({'low': 0, 'medium': 1, 'high': 2})
row_data['sleep_quality'] = row_data['sleep_quality'].map({'poor': 0, 'average': 1, 'good': 2})
row_data['internet_access'] = row_data['internet_access'].map({'no': 0, 'yes': 1})
row_data['course'] = row_data['course'].map({'b.sc': 0, 'diploma': 1, 'bca': 2, 'b.com': 3, 'ba': 4, 'bba': 5, 'b.tech': 6})

# Удаляем ненужные колонки (как при обучении)
columns_to_drop = ['id', 'age', 'gender', 'course', 'internet_access', 'exam_difficulty']
data_clean = row_data.drop(columns=columns_to_drop, axis=1)

# ========== 4. МАСШТАБИРОВАНИЕ (ТАК ЖЕ, КАК В ОБУЧЕНИИ) ==========
# 1. Список признаков для масштабирования
features_to_scale = ['study_hours', 'class_attendance', 'sleep_hours',
                     'sleep_quality', 'study_method', 'facility_rating']

# 2. Создаем копию датафрейма, чтобы не испортить оригинал
data_scaled = data_clean.copy()

# 3. Применяем StandardScaler только к выбранным столбцам
scale = StandardScaler()
data_scaled[features_to_scale] = scale.fit_transform(data_clean[features_to_scale])


# ========== 5. ПРЕДСКАЗАНИЕ ==========
predictions = voting_ensemble.predict(data_scaled)

# ========== 6. СОЗДАНИЕ SUBMISSION ==========
submission = pd.DataFrame({
    'id': passenger_ids,
    'exam_score': predictions
})

submission.to_csv('submission.csv', index=False)
print("✅ submission.csv создан")
print(submission.head())
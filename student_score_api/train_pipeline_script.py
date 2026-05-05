import pandas as pd
from app.pipeline import train_pipeline, save_pipeline

# Загружаем данные
df = pd.read_csv('/Users/dmitrii/PycharmProjects/Students_score/data/playground-series-s6e1/train.csv')

# ========== ПРЕДОБРАБОТКА (как в твоем EDA) ==========
# Копируем, чтобы не портить оригинал
df = df.copy()

# Заменяем текстовые категории на числа
df['gender'] = df['gender'].map({'female': 0, 'male': 1, 'other': 2})
df['study_method'] = df['study_method'].map({'online videos': 0, 'self-study': 1, 'coaching': 2, 'group study': 3, 'mixed': 4})
df['exam_difficulty'] = df['exam_difficulty'].map({'easy': 0, 'moderate': 1, 'hard': 2})
df['facility_rating'] = df['facility_rating'].map({'low': 0, 'medium': 1, 'high': 2})
df['sleep_quality'] = df['sleep_quality'].map({'poor': 0, 'average': 1, 'good': 2})
df['internet_access'] = df['internet_access'].map({'no': 0, 'yes': 1})
df['course'] = df['course'].map({'b.sc': 0, 'diploma': 1, 'bca': 2, 'b.com': 3, 'ba': 4, 'bba': 5, 'b.tech': 6})

# Удаляем ненужные колонки
columns_to_drop = ['id', 'age', 'gender', 'course', 'internet_access', 'exam_difficulty']
X = df.drop(columns=columns_to_drop + ['exam_score'], axis=1)
y = df['exam_score']

print("✅ Данные загружены и предобработаны")
print(f"Форма X: {X.shape}")
print(f"Колонки: {X.columns.tolist()}")
print(f"Типы данных:\n{X.dtypes}")

# Обучаем пайплайн
pipeline = train_pipeline(X, y)

# Сохраняем
save_pipeline(pipeline)
print("✅ Pipeline обучен и сохранен. Теперь он сам умеет масштабировать числа.")
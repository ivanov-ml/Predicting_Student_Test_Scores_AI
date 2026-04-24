import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

path_to_row_data = '/Users/dmitrii/PycharmProjects/Students_score/data/playground-series-s6e1/test.csv'
row_data = pd.read_csv(path_to_row_data)
#print(row_data['gender'].unique())#все уникальные значения из столбца пол
row_data['gender'].replace({'female': 0, 'male': 1, 'other':2},  inplace=True)#заменяем на числа
#print(row_data['study_method'].unique())#все уникальные значения из столбца метод обучения
row_data['study_method'].replace({'online videos': 0, 'self-study': 1, 'coaching':2, 'group study':3, 'mixed':4},  inplace=True)#заменяем на числа
#print(row_data['exam_difficulty'].unique())#Все уникальные значения из столбца сложность экзамена
row_data['exam_difficulty'].replace({'easy': 0, 'moderate': 1, 'hard': 2},  inplace=True)#заменяем на числа
#print(row_data['facility_rating'].unique())#все уникальные значения из столбца рейтинг учебного центра
row_data['facility_rating'].replace({'low': 0, 'medium': 1, 'high': 2},  inplace=True)#заменяем на числа
#print(row_data['sleep_quality'].unique())#все уникальные значения из столбца качество сна
row_data['sleep_quality'].replace({'poor': 0, 'average': 1, 'good': 2},  inplace=True)#заменяем на числа
#print(row_data['internet_access'].unique())#все уникальные значения из столбца доступ в интернет
row_data['internet_access'].replace({'no': 0, 'yes': 1},  inplace=True)#заменяем на числа
#print(row_data['course'].unique())#все уникальные значения из столбца курс
row_data['course'].replace({'b.sc': 0, 'diploma': 1, 'bca' : 2, 'b.com' : 3, 'ba': 4, 'bba': 5, 'b.tech' : 6},  inplace=True)#заменяем на числа

#print(row_data.columns)
corr_matrix = row_data.corr()# матрица корреляции по всем данным
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')#тепловая карта
#plt.show()

data_clean = row_data.drop(['id', 'age', 'gender', 'course', 'internet_access', 'exam_difficulty'], axis=1)#убрали маловажные признаки

#print(data_clean.columns)
corr_matrix_clean = data_clean.corr()# матрица корреляции по самым важным данным
sns.heatmap(corr_matrix_clean, annot=True, cmap='coolwarm')#тепловая карта
#plt.show()


# 1. Список признаков для масштабирования
features_to_scale = ['study_hours', 'class_attendance', 'sleep_hours',
                     'sleep_quality', 'study_method', 'facility_rating']

# 2. Создаем копию датафрейма, чтобы не испортить оригинал
data_scaled = data_clean.copy()

# 3. Применяем StandardScaler только к выбранным столбцам
scale = StandardScaler()
data_scaled[features_to_scale] = scale.fit_transform(data_clean[features_to_scale])

# Теперь в data_scaled признаки нормализованы, а exam_score остался прежним
print(data_scaled)

#print(data_clean_scaled.isna().sum().sum())# в данных отсутствуют пропуски


# Сохранение в CSV
data_scaled.to_csv('test_data_cleaned_scaled.csv', index=False)












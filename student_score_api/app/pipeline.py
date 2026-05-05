import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb

# ВСЕ признаки теперь числовые
NUMERICAL_FEATURES = [
    'study_hours', 'class_attendance', 'sleep_hours',
    'sleep_quality', 'study_method', 'facility_rating'
]

def create_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ('scale', StandardScaler(), NUMERICAL_FEATURES)
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', lgb.LGBMRegressor(n_estimators=100, random_state=42))
    ])
    return pipeline

def train_pipeline(X_train, y_train):
    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)
    return pipeline

def save_pipeline(pipeline, path='models/student_pipeline.pkl'):
    joblib.dump(pipeline, path)
    print(f"✅ Pipeline saved to {path}")

def load_pipeline(path='models/student_pipeline.pkl'):
    return joblib.load(path)
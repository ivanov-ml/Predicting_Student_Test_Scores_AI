import joblib
import pandas as pd


class ScorePredictor:
    def __init__(self, model_path, scaler_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict(self, input_data: dict) -> float:
        # Преобразуем входной словарь в DataFrame
        df = pd.DataFrame([input_data])

        # Масштабируем нужные фичи (те же, что и при обучении!)
        features_to_scale = ['study_hours', 'class_attendance', 'sleep_hours',
                             'sleep_quality', 'study_method', 'facility_rating']
        df[features_to_scale] = self.scaler.transform(df[features_to_scale])

        # Предсказываем
        prediction = self.model.predict(df)[0]
        return round(prediction, 2)


# Создаем глобальный объект для всего приложения
predictor = ScorePredictor('models/LGBMRegressor.pkl', 'models/scaler.pkl')
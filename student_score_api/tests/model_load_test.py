import os
import pytest
import joblib
import pytest
from sklearn.base import BaseEstimator

def test_model_file_exists():
    model_path = "models/LGBMRegressor.pkl"
    assert os.path.exists(model_path), f"Файл модели не найден: {model_path}"

@pytest.fixture
def loaded_model():
    model_path = os.path.join("models", "LGBMRegressor.pkl")
    model = joblib.load(model_path)
    return model

def test_model_type(loaded_model):
    assert isinstance(loaded_model, BaseEstimator), "Загруженный объект не является моделью sklearn"

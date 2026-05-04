import numpy as np
import pytest
from model_load_test import loaded_model


# Используем фикстуру из предыдущего примера
def test_model_prediction_shape(loaded_model):
    # Создаем фиктивные входные данные
    dummy_input = np.array([[10, 10, 10, 1, 3, 1]])

    # Делаем прогноз
    prediction = loaded_model.predict(dummy_input)

    # Проверяем форму выхода
    assert prediction.shape == (1,), "Неверная форма прогноза"
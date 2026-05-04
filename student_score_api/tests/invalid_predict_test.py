import pytest
import numpy as np
from model_load_test import loaded_model


def test_invalid_input_shape(loaded_model):
    """Проверяем, что модель выбрасывает ошибку при неправильной форме входных данных"""
    # Слишком много признаков (должно быть 6)
    invalid_input = np.array([[10, 10, 10, 1, 3, 1, 999]])

    with pytest.raises(ValueError):
        loaded_model.predict(invalid_input)


def test_invalid_input_type(loaded_model):
    """Проверяем, что модель выбрасывает ошибку при передаче строк вместо чисел"""
    invalid_input = np.array([["a", "b", "c", "d", "e", "f"]])

    with pytest.raises(Exception):  # Ловим любое исключение (ValueError, TypeError)
        loaded_model.predict(invalid_input)


def test_extremely_large_input(loaded_model):
    """Проверяем, что модель не падает на экстремально больших значениях"""
    invalid_input = np.array([[1e8, 1e8, 1e8, 1, 3, 1]])

    # Модель должна вернуть предсказание (даже если оно очень большое), а не упасть
    prediction = loaded_model.predict(invalid_input)
    assert prediction.shape == (1,), "Предсказание должно быть скаляром"
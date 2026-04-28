FROM python:3.10-slim

WORKDIR /app

# Устанавливаем системные библиотеки, нужные для LightGBM
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY submission.py .
COPY LGBMRegressor.pkl .
COPY test.csv .
COPY scaler.pkl .

CMD ["python", "submission.py"]
FROM python:3.11-slim

WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader punkt

# Копирование файлов
COPY . /app

# Expose порт 8081
EXPOSE 8081

# Запуск API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8081"]


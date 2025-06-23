FROM python:3.11-slim

WORKDIR /app

# Установка зависимостей
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader punkt



# Копирование файлов
COPY api /app/api
COPY api/client_app.py.py /app/client_app.py.py


# Запуск API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]


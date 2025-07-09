# api/main.py
import os
import logging
import json
import uuid
import asyncio
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from corrector import TextCorrector

# Загрузка переменных окружения из .env файла
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Настройка JSON логгирования
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "service": "local-api",
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Добавляем дополнительные поля если есть
        if hasattr(record, 'task_id'):
            log_entry["task_id"] = record.task_id
        if hasattr(record, 'user_id'):
            log_entry["user_id"] = record.user_id
        if hasattr(record, 'text_length'):
            log_entry["text_length"] = record.text_length
        if hasattr(record, 'processing_time'):
            log_entry["processing_time"] = record.processing_time
        if hasattr(record, 'corrections_count'):
            log_entry["corrections_count"] = record.corrections_count
        
        return json.dumps(log_entry, ensure_ascii=False)

# Настройка логгера
logger = logging.getLogger("local_api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)

logger.info("Запуск Local API сервиса")

app = FastAPI()
corrector = TextCorrector()
tasks = {}

logger.info("TextCorrector инициализирован")

class CorrectionRequest(BaseModel):
    text: str

@app.post("/submit")
async def submit_text(req: CorrectionRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    
    # Создаем логгер с task_id для отслеживания
    task_logger = logging.LoggerAdapter(logger, {'task_id': task_id})
    
    task_logger.info("Получен запрос на коррекцию", extra={
        'text_length': len(req.text),
        'task_id': task_id
    })
    
    tasks[task_id] = {"status": "processing", "result": None}
    background_tasks.add_task(process_correction, task_id, req.text)
    
    task_logger.info("Задача добавлена в очередь обработки", extra={'task_id': task_id})
    
    return {"task_id": task_id}

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    task_logger = logging.LoggerAdapter(logger, {'task_id': task_id})
    
    result = tasks.get(task_id, {"status": "not_found"})
    
    if result["status"] == "not_found":
        task_logger.warning("Запрошен несуществующий task_id", extra={'task_id': task_id})
    else:
        task_logger.info(f"Возвращен результат со статусом: {result['status']}", extra={'task_id': task_id})
    
    return result

def process_correction(task_id: str, text: str):
    import time
    start_time = time.time()
    
    task_logger = logging.LoggerAdapter(logger, {'task_id': task_id})
    
    try:
        task_logger.info("Начало обработки текста через локальную модель", extra={
            'task_id': task_id,
            'text_length': len(text)
        })
        
        corrected_text, corrections = corrector.correct(text)
        precision = corrector.calculate_precision(corrections, corrector.typos)
        
        processing_time = time.time() - start_time
        
        tasks[task_id] = {
            "status": "done",
            "result": {
                "corrected_text": corrected_text,
                "corrections": corrections,
                "precision": precision
            }
        }
        
        task_logger.info("Обработка завершена успешно", extra={
            'task_id': task_id,
            'processing_time': processing_time,
            'corrections_count': len(corrections),
            'precision': precision
        })
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        tasks[task_id] = {"status": "error", "result": str(e)}
        
        task_logger.error("Ошибка при обработке текста", extra={
            'task_id': task_id,
            'error': str(e),
            'processing_time': processing_time
        })

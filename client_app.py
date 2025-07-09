import os
import json
import logging
import requests
import time
import gradio as gr
from datetime import datetime

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
            "service": "simple-client-app",
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Добавляем дополнительные поля если есть
        if hasattr(record, 'task_id'):
            log_entry["task_id"] = record.task_id
        if hasattr(record, 'request_id'):
            log_entry["request_id"] = record.request_id
        if hasattr(record, 'text_length'):
            log_entry["text_length"] = record.text_length
        if hasattr(record, 'processing_time'):
            log_entry["processing_time"] = record.processing_time
        if hasattr(record, 'status_code'):
            log_entry["status_code"] = record.status_code
        
        return json.dumps(log_entry, ensure_ascii=False)

# Настройка логгера
logger = logging.getLogger("simple_client_app")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)

logger.info("Запуск Simple Client App")

# Получение URL API из переменных окружения
API_URL = os.getenv("LOCAL_API_URL", "http://localhost:8081")

logger.info("Настройка API URL", extra={
    'api_url': API_URL
})


def correct_text(text):
    import uuid
    request_id = str(uuid.uuid4())
    
    logger.info("Отправка запроса на коррекцию", extra={
        'text_length': len(text),
        'request_id': request_id
    })
    
    start_time = time.time()
    
    try:
        response = requests.post(f"{API_URL}/submit", json={"text": text})
        
        if response.status_code != 200:
            logger.error("Ошибка при отправке запроса", extra={
                'status_code': response.status_code,
                'request_id': request_id
            })
            return f"Ошибка при отправке задачи: {response.status_code}"
        
        task_id = response.json().get("task_id")
        if not task_id:
            logger.error("Не получен task_id", extra={
                'request_id': request_id,
                'response': response.text
            })
            return "Ошибка при отправке задачи"
        
        logger.info("Получен task_id, ожидание результата", extra={
            'task_id': task_id,
            'request_id': request_id
        })
        
        for attempt in range(30):
            response = requests.get(f"{API_URL}/result/{task_id}")
            status = response.json().get("status")
            
            if status == "done":
                processing_time = time.time() - start_time
                result = response.json().get("result")
                corrected_text = result.get("corrected_text")
                corrections = result.get("corrections")
                precision = result.get("precision")
                # Формируем красивый вывод
                corrections_str = "\n".join(f"{k} → {v}" for k, v in corrections.items())
                
                logger.info("Запрос успешно обработан", extra={
                    'task_id': task_id,
                    'request_id': request_id,
                    'processing_time': processing_time,
                    'corrections_count': len(corrections),
                    'result_length': len(corrected_text),
                    'precision': precision
                })
                
                return f"Исправленный текст:\n{corrected_text}\n\nСписок замен:\n{corrections_str}\n\nPrecision: {precision:.2f}"
            elif status == "error":
                logger.error("Ошибка обработки на сервере", extra={
                    'task_id': task_id,
                    'request_id': request_id,
                    'error': response.json().get('result')
                })
                return f"Ошибка: {response.json().get('result')}"
            
            time.sleep(1)
        
        logger.warning("Timeout при ожидании результата", extra={
            'task_id': task_id,
            'request_id': request_id,
            'timeout': 30
        })
        
        return "Время ожидания истекло"
        
    except Exception as e:
        logger.error("Исключение при обработке запроса", extra={
            'request_id': request_id,
            'error': str(e)
        })
        return f"Ошибка: {str(e)}"



iface = gr.Interface(
    fn=correct_text,
    inputs=gr.Textbox(lines=10, placeholder="Введите текст с ошибками..."),
    outputs="text",
    title="Корректор текста",
    description="Введите текст с ошибками и получите исправленный вариант"
)

if __name__ == "__main__":
    logger.info("Запуск Gradio интерфейса", extra={
        'interface_type': 'simple',
        'server_port': 7860
    })
    
    iface.launch(server_port=7860)


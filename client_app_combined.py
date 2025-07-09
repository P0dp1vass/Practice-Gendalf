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
            "service": "client-app",
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
        if hasattr(record, 'api_type'):
            log_entry["api_type"] = record.api_type
        if hasattr(record, 'status_code'):
            log_entry["status_code"] = record.status_code
        
        return json.dumps(log_entry, ensure_ascii=False)

# Настройка логгера
logger = logging.getLogger("client_app")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)

logger.info("Запуск Combined Client App")

# Получение URL API из переменных окружения
LOCAL_API_URL = os.getenv("LOCAL_API_URL", "http://localhost:8081")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "http://localhost:8082")

logger.info("Настройка API URLs", extra={
    'local_api_url': LOCAL_API_URL,
    'openai_api_url': OPENAI_API_URL
})

timeout = 120

def correct_text_local(text):
    import uuid
    request_id = str(uuid.uuid4())
    
    logger.info("Отправка запроса в локальную модель", extra={
        'api_type': 'local',
        'text_length': len(text),
        'request_id': request_id
    })
    
    start_time = time.time()
    
    try:
        response = requests.post(f"{LOCAL_API_URL}/submit", json={"text": text})
        
        if response.status_code != 200:
            logger.error("Ошибка при отправке запроса", extra={
                'api_type': 'local',
                'status_code': response.status_code,
                'request_id': request_id
            })
            return f"Ошибка при отправке задачи: {response.status_code}"
        
        task_id = response.json().get("task_id")
        if not task_id:
            logger.error("Не получен task_id", extra={
                'api_type': 'local',
                'request_id': request_id,
                'response': response.text
            })
            return "Ошибка при отправке задачи (локальная модель)"
        
        logger.info("Получен task_id, ожидание результата", extra={
            'api_type': 'local',
            'task_id': task_id,
            'request_id': request_id
        })
        
        for attempt in range(timeout):
            response = requests.get(f"{LOCAL_API_URL}/result/{task_id}")
            status = response.json().get("status")
            
            if status == "done":
                processing_time = time.time() - start_time
                result = response.json().get("result")
                corrected_text = result.get("corrected_text")
                corrections = result.get("corrections")
                precision = result.get("precision", 0)
                corrections_str = "\n".join(f"{k} → {v}" for k, v in corrections.items())
                
                logger.info("Запрос успешно обработан", extra={
                    'api_type': 'local',
                    'task_id': task_id,
                    'request_id': request_id,
                    'processing_time': processing_time,
                    'corrections_count': len(corrections),
                    'result_length': len(corrected_text),
                    'precision': precision
                })
                
                return f"Исправленный текст (локальная модель):\n{corrected_text}\n"
            elif status == "error":
                logger.error("Ошибка обработки на сервере", extra={
                    'api_type': 'local',
                    'task_id': task_id,
                    'request_id': request_id,
                    'error': response.json().get('result')
                })
                return f"Ошибка (локальная модель): {response.json().get('result')}"
            
            time.sleep(1)
        
        logger.warning("Timeout при ожидании результата", extra={
            'api_type': 'local',
            'task_id': task_id,
            'request_id': request_id,
            'timeout': timeout
        })
        
        return "Время ожидания истекло (локальная модель)"
        
    except Exception as e:
        logger.error("Исключение при обработке запроса", extra={
            'api_type': 'local',
            'request_id': request_id,
            'error': str(e)
        })
        return f"Ошибка: {str(e)}"

def correct_text_openai(text):
    import uuid
    request_id = str(uuid.uuid4())
    
    logger.info("Отправка запроса в OpenAI API", extra={
        'api_type': 'openai',
        'text_length': len(text),
        'request_id': request_id
    })
    
    start_time = time.time()
    
    try:
        response = requests.post(f"{OPENAI_API_URL}/submit", json={"text": text})
        
        if response.status_code != 200:
            logger.error("Ошибка при отправке запроса", extra={
                'api_type': 'openai',
                'status_code': response.status_code,
                'request_id': request_id
            })
            return f"Ошибка при отправке задачи: {response.status_code}"
        
        task_id = response.json().get("task_id")
        if not task_id:
            logger.error("Не получен task_id", extra={
                'api_type': 'openai',
                'request_id': request_id,
                'response': response.text
            })
            return "Ошибка при отправке задачи (OpenAI)"
        
        logger.info("Получен task_id, ожидание результата", extra={
            'api_type': 'openai',
            'task_id': task_id,
            'request_id': request_id
        })
        
        for attempt in range(timeout):
            response = requests.get(f"{OPENAI_API_URL}/result/{task_id}")
            status = response.json().get("status")
            
            if status == "done":
                processing_time = time.time() - start_time
                result = response.json().get("result")
                corrected_text = result.get("corrected_text")
                corrections = result.get("corrections")
                corrections_str = "\n".join(f"{k} → {v}" for k, v in corrections.items())
                
                logger.info("Запрос успешно обработан", extra={
                    'api_type': 'openai',
                    'task_id': task_id,
                    'request_id': request_id,
                    'processing_time': processing_time,
                    'corrections_count': len(corrections),
                    'result_length': len(corrected_text)
                })
                
                return f"Исправленный текст (OpenAI):\n{corrected_text}\n\n"
            elif status == "error":
                logger.error("Ошибка обработки на сервере", extra={
                    'api_type': 'openai',
                    'task_id': task_id,
                    'request_id': request_id,
                    'error': response.json().get('result')
                })
                return f"Ошибка (OpenAI): {response.json().get('result')}"
            
            time.sleep(1)
        
        logger.warning("Timeout при ожидании результата", extra={
            'api_type': 'openai',
            'task_id': task_id,
            'request_id': request_id,
            'timeout': timeout
        })
        
        return "Время ожидания истекло (OpenAI)"
        
    except Exception as e:
        logger.error("Исключение при обработке запроса", extra={
            'api_type': 'openai',
            'request_id': request_id,
            'error': str(e)
        })
        return f"Ошибка: {str(e)}"

def process_text(text, model_choice):
    import uuid
    session_id = str(uuid.uuid4())
    
    logger.info("Обработка пользовательского запроса", extra={
        'session_id': session_id,
        'model_choice': model_choice,
        'text_length': len(text) if text else 0
    })
    
    if not text or not text.strip():
        logger.warning("Пустой текст от пользователя", extra={
            'session_id': session_id,
            'model_choice': model_choice
        })
        return "Введите текст для обработки"
    
    if model_choice == "Локальная модель":
        logger.info("Выбрана локальная модель", extra={
            'session_id': session_id,
            'model_choice': model_choice
        })
        return correct_text_local(text)
    elif model_choice == "OpenAI":
        logger.info("Выбрана OpenAI модель", extra={
            'session_id': session_id,
            'model_choice': model_choice
        })
        return correct_text_openai(text)
    else:
        logger.warning("Модель не выбрана", extra={
            'session_id': session_id,
            'model_choice': model_choice
        })
        return "Выберите модель"

iface = gr.Interface(
    fn=process_text,
    inputs=[
        gr.Textbox(lines=10, placeholder="Введите текст с ошибками..."),
        gr.Radio(choices=["Локальная модель", "OpenAI"], label="Выберите модель")
    ],
    outputs="text",
    title="Корректор текста с выбором модели",
    description="Введите текст и выберите, какую модель использовать для коррекции"
)

if __name__ == "__main__":
    logger.info("Запуск Gradio интерфейса", extra={
        'interface_type': 'combined',
        'server_name': '0.0.0.0',
        'server_port': 7860
    })
    
    iface.launch(server_name="0.0.0.0", server_port=7860)

import os
import logging
import json
import uuid
import asyncio
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from openai import AsyncOpenAI

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
            "service": "openai-api",
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
        
        return json.dumps(log_entry, ensure_ascii=False)

# Настройка логгера
logger = logging.getLogger("openai_api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)

logger.info("Запуск OpenAI API сервиса")

app = FastAPI()

# Получение API ключа из переменных окружения
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("OPENAI_API_KEY не найден в переменных окружения")
    raise ValueError("OPENAI_API_KEY не найден в переменных окружения")

client = AsyncOpenAI(api_key=openai_api_key)
logger.info("OpenAI клиент инициализирован")

tasks = {}

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

def build_prompt(text: str) -> str:
    correct_words = [
        "гендальф", "алладин", "старбакс", "найк", "тесла", "майкрософт", "якитория",
        "1С", "бизнес", "прокачка", "сопровождение", "внедрение", "маркетинг",
        "фреш", "IT", "ЮФО", "ERP", "УНФ", "CRM", "Розница", "Лицензии",
        "Битрикс", "Стахановец", "УСН", "РСВ", "НДС", "6-НДФЛ", "3-НДФЛ",
        "4-ФСС", "ИТС", "консалтинг", "SEO", "верстка", "Wix", "Tilda",
        "OpenCart", "API", "Энтерпрайз", "WordPress", "аутсорсинг", "госсектор",
        "ЭДО", "ЭЦП", "СПАРК", "Фреш", "Контрагент", "Коннект", "Линк", "РПД", "UMI",
        "маркетплейс"
    ]

    typos = {
        "гендальс": "гендальф",
        "гендольф": "гендальф",
        "хендальф": "гендальф",
        "\"гендальс\",": "гендальф,",
        "\"гендольф\"": "гендальф",
        "\"хендальф\".": "гендальф.",
        "компния": "компания",
        "напровление": "направление",
        "сопрождения": "сопровождения",
        "якиротия,": "якитория",
        "унф": "УНФ",
        "tilda,": "Tilda",
        "токже": "также",
        "настрайкой": "настройкой",
        "тесло": "тесла",
        "несмотря": "«Несмотря",
        "отметоть,": "отметить,",
        "запск": "запуск",
        "маркетплейса": "маркетплейса",
    }

    prompt = f"""
Ты профессиональный корректор русского языка, текст, который тебе отправили, содержит ошибки из-за
помех, шумов и просто ошибок в словах людей.

Исправь ошибки в тексте с учётом следующих правил:

1. Используй следующие правильные слова и термины, не исправляй их:
{', '.join(correct_words)}

2. Исправляй следующие частые опечатки согласно словарю:
{', '.join([f'"{k}" → "{v}"' for k, v in typos.items()])}

3. Верни результат строго в формате JSON с полями:
{{
  "corrected_text": "...",  # исправленный текст
  "corrections": {{"оригинал": "исправление"}},  # словарь замен
}}

Текст для исправления:
\"\"\"{text}\"\"\"
"""
    return prompt

async def process_correction(task_id: str, text: str):
    import time
    start_time = time.time()
    
    task_logger = logging.LoggerAdapter(logger, {'task_id': task_id})
    
    try:
        task_logger.info("Начало обработки текста через OpenAI", extra={
            'task_id': task_id,
            'text_length': len(text)
        })
        
        prompt = build_prompt(text)
        
        task_logger.info("Отправка запроса в OpenAI", extra={
            'task_id': task_id,
            'prompt_length': len(prompt)
        })
        
        response = await client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        content = response.choices[0].message.content
        
        task_logger.info("Получен ответ от OpenAI", extra={
            'task_id': task_id,
            'response_length': len(content)
        })

        try:
            result = json.loads(content)
            task_logger.info("JSON ответ успешно декодирован", extra={
                'task_id': task_id,
                'corrections_count': len(result.get('corrections', {}))
            })
        except json.JSONDecodeError as e:
            task_logger.error("Ошибка декодирования JSON ответа", extra={
                'task_id': task_id,
                'error': str(e),
                'raw_content': content[:200] + "..." if len(content) > 200 else content
            })
            result = {
                "corrected_text": text,
                "corrections": {},
            }

        processing_time = time.time() - start_time
        
        tasks[task_id] = {
            "status": "done",
            "result": result
        }
        
        task_logger.info("Обработка завершена успешно", extra={
            'task_id': task_id,
            'processing_time': processing_time,
            'corrections_count': len(result.get('corrections', {}))
        })
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        tasks[task_id] = {"status": "error", "result": str(e)}
        
        task_logger.error("Ошибка при обработке текста", extra={
            'task_id': task_id,
            'error': str(e),
            'processing_time': processing_time
        })



import os
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import uuid
import json
from openai import OpenAI

# Загрузка переменных окружения из .env файла
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

print('OpenAI')
app = FastAPI()

# Получение API ключа из переменных окружения
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY не найден в переменных окружения")

client = OpenAI(api_key=openai_api_key)

tasks = {}

class CorrectionRequest(BaseModel):
    text: str

@app.post("/submit")
async def submit_text(req: CorrectionRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "processing", "result": None}
    background_tasks.add_task(process_correction, task_id, req.text)
    return {"task_id": task_id}

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    return tasks.get(task_id, {"status": "not_found"})

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

def process_correction(task_id: str, text: str):
    try:
        prompt = build_prompt(text)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )
        content = response.choices[0].message.content

        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            result = {
                "corrected_text": text,
                "corrections": {},
            }

        tasks[task_id] = {
            "status": "done",
            "result": result
        }
    except Exception as e:
        tasks[task_id] = {"status": "error", "result": str(e)}



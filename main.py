# api/main.py
import os
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import uuid
import asyncio
from corrector import TextCorrector

# Загрузка переменных окружения из .env файла
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = FastAPI()
corrector = TextCorrector()
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

def process_correction(task_id: str, text: str):
    try:
        corrected_text, corrections = corrector.correct(text)
        precision = corrector.calculate_precision(corrections, corrector.typos)
        tasks[task_id] = {
            "status": "done",
            "result": {
                "corrected_text": corrected_text,
                "corrections": corrections,
                "precision": precision
            }
        }
    except Exception as e:
        tasks[task_id] = {"status": "error", "result": str(e)}

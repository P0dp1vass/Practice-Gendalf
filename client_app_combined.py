import os
import requests
import time
import gradio as gr

# Загрузка переменных окружения из .env файла
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Получение URL API из переменных окружения
LOCAL_API_URL = os.getenv("LOCAL_API_URL", "http://localhost:8081")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "http://localhost:8082")

def correct_text_local(text):
    response = requests.post(f"{LOCAL_API_URL}/submit", json={"text": text})
    task_id = response.json().get("task_id")
    if not task_id:
        return "Ошибка при отправке задачи (локальная модель)"
    for _ in range(30):
        response = requests.get(f"{LOCAL_API_URL}/result/{task_id}")
        status = response.json().get("status")
        if status == "done":
            result = response.json().get("result")
            corrected_text = result.get("corrected_text")
            corrections = result.get("corrections")
            precision = result.get("precision", 0)
            corrections_str = "\n".join(f"{k} → {v}" for k, v in corrections.items())
            return f"Исправленный текст (локальная модель):\n{corrected_text}\n"
        elif status == "error":
            return f"Ошибка (локальная модель): {response.json().get('result')}"
        time.sleep(1)
    return "Время ожидания истекло (локальная модель)"

def correct_text_openai(text):
    response = requests.post(f"{OPENAI_API_URL}/submit", json={"text": text})
    task_id = response.json().get("task_id")
    if not task_id:
        return "Ошибка при отправке задачи (OpenAI)"
    for _ in range(30):
        response = requests.get(f"{OPENAI_API_URL}/result/{task_id}")
        status = response.json().get("status")
        if status == "done":
            result = response.json().get("result")
            corrected_text = result.get("corrected_text")
            corrections = result.get("corrections")
            corrections_str = "\n".join(f"{k} → {v}" for k, v in corrections.items())
            return f"Исправленный текст (OpenAI):\n{corrected_text}\n\n"
        elif status == "error":
            return f"Ошибка (OpenAI): {response.json().get('result')}"
        time.sleep(1)
    return "Время ожидания истекло (OpenAI)"

def process_text(text, model_choice):
    if not text or not text.strip():
        return "Введите текст для обработки"
    if model_choice == "Локальная модель":
        return correct_text_local(text)
    elif model_choice == "OpenAI":
        return correct_text_openai(text)
    else:
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
    iface.launch(server_port=7860)

import requests
import time
import gradio as gr

print('OpenAI')
API_URL = "http://localhost:8000"
timeout = 120

def correct_text(text):
    response = requests.post(f"{API_URL}/submit", json={"text": text})
    task_id = response.json().get("task_id")
    if not task_id:
        return "Ошибка при отправке задачи"
    for _ in range(timeout):
        response = requests.get(f"{API_URL}/result/{task_id}")
        status = response.json().get("status")
        if status == "done":
            result = response.json().get("result")
            corrected_text = result.get("corrected_text")
            corrections = result.get("corrections")
            corrections_str = "\n".join(f"{k} → {v}" for k, v in corrections.items())
            return f"Исправленный текст OpenAI:\n{corrected_text}\n\nСписок замен:\n{corrections_str}\n\n"
        elif status == "error":
            return f"Ошибка: {response.json().get('result')}"
        time.sleep(1)
    return "Время ожидания истекло"

iface = gr.Interface(
    fn=correct_text,
    inputs=gr.Textbox(lines=10, placeholder="Введите текст с ошибками..."),
    outputs="text",
    title="Корректор текста OpenAI",
    description="Введите текст с ошибками и получите исправленный вариант"
)

if __name__ == "__main__":
    iface.launch(server_port=7860)

import os
import json
import io
import sys
import uvicorn
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from dotenv import load_dotenv
from pydantic import BaseModel
from gtts import gTTS

# 1. Загрузка и настройка
load_dotenv()
raw_key = os.getenv("GOOGLE_API_KEY")
if not raw_key:
    print("ОШИБКА: Нет ключа в .env", flush=True)
    exit(1)
CLEAN_KEY = "".join(c for c in raw_key if c.isalnum() or c in "-_")
genai.configure(api_key=CLEAN_KEY)

# Модель
MODEL_NAME = 'gemini-1.5-flash-002'
model = genai.GenerativeModel(MODEL_NAME)

# --- ОТКЛЮЧАЕМ ФИЛЬТРЫ БЕЗОПАСНОСТИ (ЧТОБЫ НЕ БЛОКИРОВАЛ МЕДИЦИНУ) ---
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class DietRequest(BaseModel):
    analysis_data: dict

class ChatRequest(BaseModel):
    question: str
    analysis_data: dict

SYSTEM_PROMPT = """
Ты профессиональный медицинский ассистент-триаж.
Твоя задача: проанализировать анализ крови и направить пациента к НУЖНЫМ специалистам.

ПРАВИЛА:
1. НИКОГДА не ставь диагнозы.
2. В "priority_action" назови конкретных врачей (Эндокринолог, Кардиолог и т.д.).
3. Если отклонений нет - пиши "Терапевт (планово)".

Формат JSON:
{
  "client": { "fio": "Имя", "gender": "Пол", "age": "Возраст", "date": "Дата" },
  "abnormal_results": [ 
      { "name": "Показатель", "range": "Референс", "value": "Значение", "analysis": "Объяснение." } 
  ],
  "expert_summary": "Сводка по системам организма.",
  "priority_action": "1. СПЕЦИАЛИСТ (причина); 2. СПЕЦИАЛИСТ (причина)."
}
"""

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    print(f"\n--> [АНАЛИЗ] Получен файл: {file.filename}", flush=True)
    try:
        content = await file.read()
        response = model.generate_content(
            [SYSTEM_PROMPT, {"mime_type": file.content_type, "data": content}],
            safety_settings=SAFETY_SETTINGS
        )
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        return JSONResponse(content=json.loads(clean_text))
    except Exception as e:
        print(f"!!! ОШИБКА АНАЛИЗА: {e}", flush=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/api/diet")
async def generate_diet(request: DietRequest):
    print("\n--> [ДИЕТА] Генерирую план питания...", flush=True)
    try:
        diet_prompt = f"""
        Ты диетолог. Составь план питания на 1 день на основе анализов.
        Анализы: {json.dumps(request.analysis_data, ensure_ascii=False)}
        Задача: Компенсируй дефициты продуктами. Меню (Завтрак, Обед, Ужин). Используй эмодзи.
        """
        response = model.generate_content(diet_prompt, safety_settings=SAFETY_SETTINGS)
        return JSONResponse(content={"diet_plan": response.text})
    except Exception as e:
        print(f"!!! ОШИБКА ДИЕТЫ: {e}", flush=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/api/voice")
async def generate_voice(item: dict = Body(...)):
    print("\n--> [ГОЛОС] Синтез...", flush=True)
    try:
        text = item.get("text", "")
        if not text: return JSONResponse(content={"error": "Нет текста"}, status_code=400)
        
        tts = gTTS(text=text, lang='ru')
        audio_stream = io.BytesIO()
        tts.write_to_fp(audio_stream)
        audio_stream.seek(0)
        return StreamingResponse(audio_stream, media_type="audio/mp3")
    except Exception as e:
        print(f"!!! ОШИБКА ГОЛОСА: {e}", flush=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/api/chat")
async def chat_with_ai(request: ChatRequest):
    print(f"\n--> [ЧАТ] Вопрос: {request.question}", flush=True)
    try:
        chat_prompt = f"""
        Ты медицинский консультант Blood.AI.
        КОНТЕКСТ ПАЦИЕНТА: {json.dumps(request.analysis_data, ensure_ascii=False)}
        ВОПРОС: "{request.question}"
        Отвечай кратко, вежливо, опираясь на анализы. Не ставь диагнозы.
        """
        response = model.generate_content(chat_prompt, safety_settings=SAFETY_SETTINGS)
        return JSONResponse(content={"answer": response.text})
    except Exception as e:
        print(f"!!! ОШИБКА ЧАТА: {e}", flush=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

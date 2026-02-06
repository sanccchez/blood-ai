import os
import json
import io
import uvicorn
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from dotenv import load_dotenv
from pydantic import BaseModel
from gtts import gTTS

# 1. Загрузка настроек
load_dotenv()
raw_key = os.getenv("GOOGLE_API_KEY")

# --- БЛОК БЕЗОПАСНОСТИ КЛЮЧА ---
if not raw_key:
    # Если локально .env не сработал
    DIRECT_KEY = "PASTE_YOUR_KEY_HERE"
    if DIRECT_KEY != "PASTE_YOUR_KEY_HERE":
        raw_key = DIRECT_KEY

if raw_key:
    CLEAN_KEY = "".join(c for c in raw_key if c.isalnum() or c in "-_")
    genai.configure(api_key=CLEAN_KEY)
    print(f"--> Ключ загружен. Длина: {len(CLEAN_KEY)}", flush=True)

# Модель
MODEL_NAME = 'gemini-2.5-flash' 
model = genai.GenerativeModel(MODEL_NAME)

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

# --- ОБНОВЛЕННЫЙ ПРОМПТ (МУЛЬТИ-ВРАЧИ) ---
SYSTEM_PROMPT = """
Ты профессиональный медицинский консилиум. Твоя задача — извлечь данные в JSON.

ГЛАВНОЕ ПРАВИЛО ПО ВРАЧАМ (поле priority_action):
Ты должен перечислить ВСЕХ специалистов, к которым нужно обратиться, через запятую.
1. ТТГ, Т3, Т4, Глюкоза, Инсулин -> Эндокринолог.
2. Гемоглобин, Ферритин, Железо -> Гематолог.
3. Холестерин, Липидный профиль, Сердце -> Кардиолог.
4. АЛТ, АСТ, Билирубин -> Гастроэнтеролог.
5. Если затронуто несколько систем (например, щитовидка И холестерин) -> пиши: "Эндокринолог, Кардиолог".
6. Не пиши "Терапевт", если есть конкретные отклонения. Терапевт только для нормы.

ТВОЙ ОТВЕТ ДОЛЖЕН БЫТЬ ТОЛЬКО ВАЛИДНЫМ JSON:
{
  "client": { "fio": "Имя или Не указано", "gender": "Пол", "age": "Возраст", "date": "Дата" },
  "abnormal_results": [ 
      { "name": "Показатель", "range": "Норма", "value": "Значение", "analysis": "Краткое объяснение" } 
  ],
  "expert_summary": "Сводка по здоровью.",
  "priority_action": "Эндокринолог, Кардиолог" 
}
"""

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    print(f"\n--> [АНАЛИЗ] Получен файл: {file.filename}", flush=True)
    try:
        content = await file.read()
        
        # Запрос в Google
        response = model.generate_content(
            [SYSTEM_PROMPT, {"mime_type": file.content_type, "data": content}],
            safety_settings=SAFETY_SETTINGS,
            generation_config={"response_mime_type": "application/json"}
        )
        
        try:
            text_response = response.text.replace("```json", "").replace("```", "").strip()
            data = json.loads(text_response)
            
            # Подушка безопасности
            if "client" not in data: data["client"] = {}
            if "abnormal_results" not in data: data["abnormal_results"] = []
            if "expert_summary" not in data: data["expert_summary"] = "Анализ завершен."
            if "priority_action" not in data: data["priority_action"] = "Терапевт"
                
            return JSONResponse(content=data)

        except json.JSONDecodeError:
            print("!!! ОШИБКА JSON")
            return JSONResponse(content={
                "client": {"fio": "Ошибка", "gender": "-", "age": "-", "date": "-"},
                "abnormal_results": [],
                "expert_summary": "Ошибка обработки.",
                "priority_action": "Повторить"
            })

    except Exception as e:
        err_msg = str(e)
        print(f"!!! ОШИБКА: {err_msg}", flush=True)
        if "429" in err_msg:
             return JSONResponse(content={"error": "Слишком много запросов. Подождите 1 минуту."}, status_code=429)
        return JSONResponse(content={"error": err_msg}, status_code=500)

@app.post("/api/diet")
async def generate_diet(request: DietRequest):
    try:
        prompt = f"Составь меню на 1 день для: {json.dumps(request.analysis_data, ensure_ascii=False)}. HTML теги <b>,<br>."
        response = model.generate_content(prompt)
        return JSONResponse(content={"diet_plan": response.text})
    except Exception as e:
        return JSONResponse(content={"diet_plan": "Ошибка диеты."}, status_code=200)

@app.post("/api/voice")
async def generate_voice(item: dict = Body(...)):
    try:
        text = item.get("text", "Нет данных")
        tts = gTTS(text=text, lang='ru')
        audio_stream = io.BytesIO()
        tts.write_to_fp(audio_stream)
        audio_stream.seek(0)
        return StreamingResponse(audio_stream, media_type="audio/mp3")
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/api/chat")
async def chat_with_ai(request: ChatRequest):
    try:
        chat_prompt = f"Контекст: {json.dumps(request.analysis_data, ensure_ascii=False)}. Вопрос: {request.question}"
        response = model.generate_content(chat_prompt)
        return JSONResponse(content={"answer": response.text})
    except Exception as e:
        return JSONResponse(content={"answer": "Ошибка чата."}, status_code=200)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)

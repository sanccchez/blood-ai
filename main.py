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

# Модель (оставляем 2.5, она у вас работает)
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

# --- ОБНОВЛЕННЫЙ ПРОМПТ (СТРОГИЙ ВЫБОР ВРАЧА) ---
SYSTEM_PROMPT = """
Ты профессиональный медицинский ассистент. Твоя задача — извлечь данные из анализа крови в строгий JSON.

ВАЖНЫЕ ПРАВИЛА ПО ВРАЧАМ (поле priority_action):
1. Если есть отклонения в ТТГ, Т3, Т4, Глюкозе, Инсулине -> пиши "Эндокринолог".
2. Если Гемоглобин, Ферритин, Железо, Эритроциты не в норме -> пиши "Гематолог".
3. Если Холестерин, ЛПНП, Триглицериды -> пиши "Кардиолог".
4. Если Билирубин, АЛТ, АСТ -> пиши "Гастроэнтеролог".
5. Пиши "Терапевт" ТОЛЬКО если все анализы в норме или отклонения минимальны.
6. Пиши ТОЛЬКО специальность (одно-два слова), без лишних фраз.

ТВОЙ ОТВЕТ ДОЛЖЕН БЫТЬ ТОЛЬКО ВАЛИДНЫМ JSON СЛЕДУЮЩЕЙ СТРУКТУРЫ:
{
  "client": { "fio": "Имя или Не указано", "gender": "Пол", "age": "Возраст", "date": "Дата" },
  "abnormal_results": [ 
      { "name": "Показатель", "range": "Норма", "value": "Значение", "analysis": "Краткое объяснение простым языком" } 
  ],
  "expert_summary": "Сводка по здоровью (2-3 предложения).",
  "priority_action": "Специальность врача"
}
"""

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    print(f"\n--> [АНАЛИЗ] Получен файл: {file.filename}", flush=True)
    try:
        content = await file.read()
        
        # Запрос в Google с требованием JSON
        response = model.generate_content(
            [SYSTEM_PROMPT, {"mime_type": file.content_type, "data": content}],
            safety_settings=SAFETY_SETTINGS,
            generation_config={"response_mime_type": "application/json"}
        )
        
        try:
            # Очистка от markdown на всякий случай
            text_response = response.text.replace("```json", "").replace("```", "").strip()
            data = json.loads(text_response)
            
            # Подушка безопасности (если поля пустые)
            if "client" not in data: data["client"] = {}
            if "abnormal_results" not in data: data["abnormal_results"] = []
            if "expert_summary" not in data: data["expert_summary"] = "Анализ завершен."
            if "priority_action" not in data: data["priority_action"] = "Терапевт"
                
            return JSONResponse(content=data)

        except json.JSONDecodeError:
            print("!!! ОШИБКА JSON: Модель вернула некорректный формат.")
            return JSONResponse(content={
                "client": {"fio": "Ошибка чтения", "gender": "-", "age": "-", "date": "-"},
                "abnormal_results": [],
                "expert_summary": "Произошла ошибка обработки данных ИИ. Попробуйте еще раз.",
                "priority_action": "Повторить загрузку"
            })

    except Exception as e:
        err_msg = str(e)
        print(f"!!! КРИТИЧЕСКАЯ ОШИБКА: {err_msg}", flush=True)
        if "429" in err_msg:
             return JSONResponse(content={"error": "Слишком много запросов. Подождите 1 минуту (Лимит Google)."}, status_code=429)
        return JSONResponse(content={"error": err_msg}, status_code=500)

@app.post("/api/diet")
async def generate_diet(request: DietRequest):
    try:
        prompt = f"Составь диету на 1 день для пациента с такими анализами: {json.dumps(request.analysis_data, ensure_ascii=False)}. Используй HTML теги <b> и <br>."
        response = model.generate_content(prompt)
        return JSONResponse(content={"diet_plan": response.text})
    except Exception as e:
        return JSONResponse(content={"diet_plan": "Не удалось составить диету."}, status_code=200)

@app.post("/api/voice")
async def generate_voice(item: dict = Body(...)):
    try:
        text = item.get("text", "Нет данных для озвучки")
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
        chat_prompt = f"Контекст анализов: {json.dumps(request.analysis_data, ensure_ascii=False)}. Вопрос: {request.question}"
        response = model.generate_content(chat_prompt)
        return JSONResponse(content={"answer": response.text})
    except Exception as e:
        return JSONResponse(content={"answer": "Извините, я сейчас не могу ответить."}, status_code=200)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)

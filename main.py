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

# 1. Загрузка
load_dotenv()
raw_key = os.getenv("GOOGLE_API_KEY")

# --- БЛОК ОЧИСТКИ КЛЮЧА ---
if not raw_key:
    # Если ключа нет в .env, пробуем хардкод (на случай локальных тестов)
    # Вставьте ключ сюда, если .env не работает
    DIRECT_KEY = "PASTE_YOUR_KEY_HERE"
    if DIRECT_KEY != "PASTE_YOUR_KEY_HERE":
        raw_key = DIRECT_KEY
    else:
        print("ОШИБКА: Нет ключа API", flush=True)

if raw_key:
    CLEAN_KEY = "".join(c for c in raw_key if c.isalnum() or c in "-_")
    genai.configure(api_key=CLEAN_KEY)
    print(f"--> Ключ загружен. Длина: {len(CLEAN_KEY)}", flush=True)

# Модель (оставляем 2.5, раз она у вас работает)
MODEL_NAME = 'gemini-2.5-flash' 
model = genai.GenerativeModel(MODEL_NAME)

# Настройки безопасности
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

# Промпт стал строже
SYSTEM_PROMPT = """
Ты профессиональный медицинский ассистент. Твоя задача — извлечь данные в JSON.

ВАЖНЫЕ ПРАВИЛА ПО ВРАЧАМ:
1. Если видишь отклонения в гормонах (ТТГ, Т4) -> пиши "Эндокринолог".
2. Если железо/ферритин/гемоглобин -> пиши "Гематолог".
3. Если холестерин/сердце -> пиши "Кардиолог".
4. Пиши "Терапевт" ТОЛЬКО если все анализы в норме или отклонения незначительны.
5. В поле "priority_action" пиши ТОЛЬКО специальность врача (без лишних слов).

Формат ответа (JSON):
{
  "client": { "fio": "Имя или Не указано", "gender": "Пол", "age": "Возраст", "date": "Дата" },
  "abnormal_results": [ 
      { "name": "Показатель", "range": "Норма", "value": "Значение", "analysis": "Кратко: что это значит" } 
  ],
  "expert_summary": "Сводка по здоровью (2-3 предложения).",
  "priority_action": "Эндокринолог"
}
"""@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    print(f"\n--> [АНАЛИЗ] Получен файл: {file.filename}", flush=True)
    try:
        content = await file.read()
        
        # --- ГЛАВНОЕ ИЗМЕНЕНИЕ: FORCED JSON ---
        # Мы заставляем модель отвечать в формате JSON
        response = model.generate_content(
            [SYSTEM_PROMPT, {"mime_type": file.content_type, "data": content}],
            safety_settings=SAFETY_SETTINGS,
            generation_config={"response_mime_type": "application/json"}
        )
        
        # Попытка парсинга
        try:
            # Очистка на случай, если модель добавит ```json
            text_response = response.text.replace("```json", "").replace("```", "").strip()
            data = json.loads(text_response)
            
            # --- ПОДУШКА БЕЗОПАСНОСТИ ---
            # Если модель забыла добавить блок client, мы добавим его сами, чтобы сайт не упал
            if "client" not in data:
                data["client"] = {"fio": "Пациент (данные не найдены)", "gender": "-", "age": "-", "date": "-"}
            if "abnormal_results" not in data:
                data["abnormal_results"] = []
            if "expert_summary" not in data:
                data["expert_summary"] = "Не удалось сформировать автоматическое заключение. Обратитесь к врачу."
            if "priority_action" not in data:
                data["priority_action"] = "Терапевт"
                
            return JSONResponse(content=data)

        except json.JSONDecodeError:
            # Если вернулся мусор вместо JSON
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
        # Обработка лимитов
        if "429" in err_msg:
             return JSONResponse(content={"error": "Слишком много запросов. Подождите 1 минуту."}, status_code=429)
        return JSONResponse(content={"error": err_msg}, status_code=500)

@app.post("/api/diet")
async def generate_diet(request: DietRequest):
    # Упрощенная логика для стабильности
    try:
        prompt = f"Составь диету на 1 день для пациента с такими анализами: {json.dumps(request.analysis_data, ensure_ascii=False)}. Используй HTML теги <b> и <br>."
        response = model.generate_content(prompt)
        return JSONResponse(content={"diet_plan": response.text})
    except Exception as e:
        return JSONResponse(content={"diet_plan": "Не удалось составить диету."}, status_code=200) # Возвращаем 200, чтобы не пугать юзера ошибкой

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
        print(f"!!! ОШИБКА ГОЛОСА: {e}")
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

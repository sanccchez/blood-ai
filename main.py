import os
import json
import uvicorn
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# 1. Загрузка настроек
load_dotenv()

# --- ВАЖНАЯ НАСТРОЙКА ---
# Мы используем "Lite" версию новейшей модели. 
# Она доступна в вашем аккаунте и должна быть лояльнее к лимитам.
MODEL_NAME = 'gemini-2.0-flash-lite'

# Получаем ключ из сервера (Environment Variables)
raw_key = os.getenv("GOOGLE_API_KEY")

if not raw_key:
    print("!!! КРИТИЧЕСКАЯ ОШИБКА: Не найден GOOGLE_API_KEY в настройках сервера Render!", flush=True)
else:
    CLEAN_KEY = "".join(c for c in raw_key if c.isalnum() or c in "-_")
    try:
        genai.configure(api_key=CLEAN_KEY)
        print(f"--> [СИСТЕМА] Ключ загружен. Пробуем модель: {MODEL_NAME}", flush=True)
    except Exception as e:
        print(f"!!! ОШИБКА КОНФИГУРАЦИИ: {e}", flush=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация модели
try:
    model = genai.GenerativeModel(MODEL_NAME)
except Exception as e:
    print(f"Ошибка создания модели: {e}")

@app.post("/api/analyze")
async def analyze_blood(file: UploadFile = File(...)):
    print(f"\n--> [ЗАПРОС] Получен файл: {file.filename}", flush=True)
    
    if not raw_key:
        return JSONResponse(content={"analysis": "Ошибка сервера: Отсутствует API ключ."}, status_code=500)

    try:
        content = await file.read()
        
        prompt = """
        Ты опытный врач-гематолог. Внимательно проанализируй этот анализ крови.
        
        ТВОЯ ЗАДАЧА:
        1. Выяви все показатели, которые выходят за пределы нормы.
        2. Объясни простым языком, о чем это говорит.
        3. Дай краткие рекомендации по питанию.
        
        ОТВЕТ: Строго в формате HTML (<b>, <ul>, <br>), без ```html.
        """

        # Отправка в Google
        response = model.generate_content(
            [prompt, {"mime_type": "application/pdf", "data": content}],
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
        )
        
        return JSONResponse(content={"analysis": response.text})

    except Exception as e:
        error_msg = str(e)
        print(f"!!! ОШИБКА ПРИ АНАЛИЗЕ: {error_msg}", flush=True)
        
        if "429" in error_msg:
            return JSONResponse(content={"analysis": "⚠️ Слишком быстро. Google просит подождать 1 минуту (Лимит студенческой версии)."}, status_code=429)
        if "404" in error_msg:
            return JSONResponse(content={"analysis": f"⚠️ Модель {MODEL_NAME} недоступна для этого ключа. Нужен ключ из другого проекта."}, status_code=404)
             
        return JSONResponse(content={"analysis": f"Ошибка: {error_msg}"}, status_code=500)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)

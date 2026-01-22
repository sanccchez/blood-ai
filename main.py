import os
import json
import uvicorn
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# 1. Загрузка переменных окружения
load_dotenv()

# --- НАСТРОЙКИ ---
# Используем 1.5 Flash, так как у неё самые большие лимиты (15 RPM)
# Новые модели (2.0/2.5) имеют лимит 2 RPM и быстро выдают ошибку 429
MODEL_NAME = 'gemini-1.5-flash'

# Получаем ключ из сервера (Environment Variables)
raw_key = os.getenv("GOOGLE_API_KEY")

if not raw_key:
    print("!!! КРИТИЧЕСКАЯ ОШИБКА: Не найден GOOGLE_API_KEY в настройках сервера Render!", flush=True)
else:
    # Очистка ключа от мусора
    CLEAN_KEY = "".join(c for c in raw_key if c.isalnum() or c in "-_")
    try:
        genai.configure(api_key=CLEAN_KEY)
        print(f"--> [СИСТЕМА] Ключ загружен. Используем модель: {MODEL_NAME}", flush=True)
    except Exception as e:
        print(f"!!! ОШИБКА КОНФИГУРАЦИИ: {e}", flush=True)

app = FastAPI()

# Разрешаем доступ с любого сайта
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
        return JSONResponse(content={"analysis": "Ошибка сервера: Отсутствует API ключ. Проверьте настройки Render."}, status_code=500)

    try:
        content = await file.read()
        
        # Инструкция для ИИ
        prompt = """
        Ты опытный врач-гематолог. Внимательно проанализируй этот анализ крови (PDF/изображение).
        
        ТВОЯ ЗАДАЧА:
        1. Выяви все показатели, которые выходят за пределы референсных значений (нормы).
        2. Объясни простым и понятным языком, о чем может говорить каждое отклонение.
        3. Дай краткие рекомендации по питанию или образу жизни для коррекции.
        4. Если все показатели в норме - поздравь пациента и напиши, что всё хорошо.
        
        ВАЖНО:
        - Ответ должен быть в формате HTML (используй <b> для жирного, <br> для переноса, <ul><li> для списков).
        - НЕ используй markdown (```html). Просто верни текст.
        - В конце добавь: "<br><br><b>Важно:</b> Это не диагноз. Обязательно проконсультируйтесь с врачом."
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
        
        # Проверка ответа
        if not response.text:
            return JSONResponse(content={"analysis": "ИИ вернул пустой ответ. Попробуйте еще раз."}, status_code=500)
            
        return JSONResponse(content={"analysis": response.text})

    except Exception as e:
        error_msg = str(e)
        print(f"!!! ОШИБКА ПРИ АНАЛИЗЕ: {error_msg}", flush=True)
        
        # Обработка популярных ошибок для вывода пользователю
        if "429" in error_msg:
            return JSONResponse(content={"analysis": "⚠️ Слишком много запросов. Google временно ограничил доступ. Пожалуйста, подождите 2-3 минуты и попробуйте снова."}, status_code=429)
        if "404" in error_msg:
            return JSONResponse(content={"analysis": "⚠️ Ошибка настройки ключа (404). Модель не найдена или ключ не имеет доступа."}, status_code=404)
        if "403" in error_msg:
             return JSONResponse(content={"analysis": "⚠️ Ошибка безопасности (403). Google заблокировал этот ключ. Нужен новый."}, status_code=403)
             
        return JSONResponse(content={"analysis": f"Техническая ошибка: {error_msg}"}, status_code=500)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)

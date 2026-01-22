import os
import json
import uvicorn
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# --- НАСТРОЙКА КЛЮЧА (Самая важная часть) ---
# Если через переменные окружения не работает, вставь ключ прямо сюда между кавычками.
# Пример: DIRECT_KEY = "AIzaSyD..."
DIRECT_KEY = "AIzaSyCAz70hCFdI-Q7KC17bJYIcgCjIkZBKXMk"

load_dotenv()
# Сначала пробуем взять ключ из кода (если ты его вставил), иначе ищем в сервере
raw_key = DIRECT_KEY if DIRECT_KEY != "PASTE_YOUR_KEY_HERE" else os.getenv("GOOGLE_API_KEY")

if not raw_key:
    print("!!! ОШИБКА: Ключ API не найден ни в коде, ни в настройках сервера.", flush=True)
    # Мы не выходим (exit), чтобы сервер не падал, но работать он не сможет без ключа
else:
    # Очистка ключа от случайных пробелов и переносов строк
    CLEAN_KEY = "".join(c for c in raw_key if c.isalnum() or c in "-_")
    genai.configure(api_key=CLEAN_KEY)
    print(f"--> Ключ принят. Длина: {len(CLEAN_KEY)} символов", flush=True)

# --- ВЫБОР МОДЕЛИ ---
# Используем самую стабильную версию. Не меняй это название.
MODEL_NAME = 'gemini-1.5-flash' 
model = genai.GenerativeModel(MODEL_NAME)

app = FastAPI()

# Разрешаем доступ сайту
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "Server is running", "model": MODEL_NAME}

@app.post("/api/analyze")
async def analyze_blood(file: UploadFile = File(...)):
    print(f"\n--> [АНАЛИЗ] Получен файл: {file.filename}", flush=True)
    
    try:
        # 1. Читаем файл
        file_content = await file.read()
        
        # 2. Формируем запрос для Gemini
        # Мы отправляем байты напрямую, указывая MIME-тип PDF
        prompt_parts = [
            {"mime_type": "application/pdf", "data": file_content},
            """
            Ты профессиональный врач-гематолог. Проанализируй этот анализ крови.
            ТВОЯ ЗАДАЧА:
            1. Выпиши все показатели, которые выходят за пределы нормы.
            2. Объясни простым языком, о чем это может говорить.
            3. Дай краткие рекомендации по питанию или образу жизни.
            4. В конце обязательно добавь фразу: "Обязательно проконсультируйтесь с врачом."
            
            Ответ дай в формате HTML (используй <b> для жирного, <ul> для списков), но БЕЗ тегов ```html и без body.
            Просто верни чистый текст разметки для вставки в div.
            """
        ]

        # 3. Отправляем в Google (с отключенными фильтрами безопасности)
        response = model.generate_content(
            prompt_parts,
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
        )
        
        # 4. Проверяем ответ
        if not response.text:
            raise ValueError("Google вернул пустой ответ.")
            
        return JSONResponse(content={"analysis": response.text})

    except Exception as e:
        error_msg = str(e)
        print(f"!!! ОШИБКА АНАЛИЗА: {error_msg}", flush=True)
        
        # Если ошибка 404 - даем понятную подсказку
        if "404" in error_msg:
            return JSONResponse(content={"analysis": "Ошибка 404: Google не видит модель. Проверьте API ключ (возможно, нужен новый проект в Google AI Studio)."}, status_code=500)
            
        return JSONResponse(content={"analysis": f"Произошла ошибка при обработке: {error_msg}"}, status_code=500)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)

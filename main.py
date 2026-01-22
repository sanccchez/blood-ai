import os
import json
import uvicorn
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# --- ВАШ КЛЮЧ ---
# (Оставьте его здесь, как есть)
DIRECT_KEY = "PASTE_YOUR_KEY_HERE"

load_dotenv()
# Логика выбора ключа
raw_key = DIRECT_KEY if DIRECT_KEY != "PASTE_YOUR_KEY_HERE" else os.getenv("GOOGLE_API_KEY")

if not raw_key:
    print("!!! ОШИБКА: Нет ключа.", flush=True)
else:
    CLEAN_KEY = "".join(c for c in raw_key if c.isalnum() or c in "-_")
    genai.configure(api_key=CLEAN_KEY)
    print(f"--> Ключ принят. Длина: {len(CLEAN_KEY)} символов", flush=True)

    # --- 🔥 ДИАГНОСТИКА: ПРОВЕРКА ДОСТУПНЫХ МОДЕЛЕЙ 🔥 ---
    print("\n--- ЗАПРОС К GOOGLE: КАКИЕ МОДЕЛИ ДОСТУПНЫ? ---", flush=True)
    try:
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"✅ Доступна: {m.name}", flush=True)
                available_models.append(m.name)
        
        if not available_models:
            print("!!! СПИСОК ПУСТ. Ключ рабочий, но API выключен или регион заблокирован.", flush=True)
        else:
            print(f"--> Всего найдено моделей: {len(available_models)}", flush=True)
    except Exception as e:
        print(f"!!! ОШИБКА ПРИ ПОЛУЧЕНИИ СПИСКА МОДЕЛЕЙ: {e}", flush=True)
    print("-----------------------------------------------\n", flush=True)

# Используем стандартную модель
MODEL_NAME = 'gemini-1.5-flash'
model = genai.GenerativeModel(MODEL_NAME)

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

@app.post("/api/analyze")
async def analyze_blood(file: UploadFile = File(...)):
    print(f"--> [АНАЛИЗ] Файл: {file.filename}", flush=True)
    try:
        content = await file.read()
        # Простой промпт для теста
        prompt = "Ты врач. Проанализируй этот файл. Ответь кратко."
        
        response = model.generate_content(
            [prompt, {"mime_type": "application/pdf", "data": content}]
        )
        return JSONResponse(content={"analysis": response.text})
    except Exception as e:
        print(f"!!! ОШИБКА: {e}", flush=True)
        return JSONResponse(content={"analysis": f"Ошибка: {str(e)}"}, status_code=500)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)

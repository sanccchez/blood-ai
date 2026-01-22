import os
import json
import uvicorn
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# --- НАСТРОЙКИ ---
# ВАЖНО: Вставь свой ключ AIza... внутрь кавычек ниже!
DIRECT_KEY = "AIzaSyCAz70hCFdI-Q7KC17bJYIcgCjIkZBKXMk"

# Берем модель 2.0 Flash, которая ТОЧНО есть в твоих логах
MODEL_NAME = 'gemini-2.0-flash'

# 1. Настройка ключа
load_dotenv()
# Если вы вставили ключ выше, используем его. Если нет - ищем в сервере.
raw_key = DIRECT_KEY if DIRECT_KEY != "PASTE_YOUR_KEY_HERE" else os.getenv("GOOGLE_API_KEY")

if not raw_key:
    print("!!! ОШИБКА: Ключ не найден. Вставьте его в строку 16 в main.py", flush=True)
    # Не выходим, чтобы сервер не падал циклично, но работать он не будет
else:
    CLEAN_KEY = "".join(c for c in raw_key if c.isalnum() or c in "-_")
    genai.configure(api_key=CLEAN_KEY)
    print(f"--> Ключ принят. Модель выбрана: {MODEL_NAME}", flush=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

model = genai.GenerativeModel(MODEL_NAME)

@app.post("/api/analyze")
async def analyze_blood(file: UploadFile = File(...)):
    print(f"\n--> [АНАЛИЗ] Получен файл: {file.filename}", flush=True)
    try:
        content = await file.read()
        
        # Промпт для врача
        prompt = """
        Ты опытный врач-гематолог. Проанализируй этот анализ крови (файл во вложении).
        
        ТВОЯ ЗАДАЧА:
        1. Найди показатели, выходящие за норму.
        2. Объясни доступно, что это значит.
        3. Дай краткие рекомендации по питанию.
        4. Если всё в норме - так и напиши.
        
        Ответь в формате HTML (теги <b>, <ul>, <br>), но без ```html.
        """

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
        err = str(e)
        print(f"!!! ОШИБКА: {err}", flush=True)
        if "404" in err:
            return JSONResponse(content={"analysis": f"Ошибка доступа к модели {MODEL_NAME}. Проверьте ключ."}, status_code=500)
        return JSONResponse(content={"analysis": f"Ошибка сервера: {err}"}, status_code=500)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)

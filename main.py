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

# Мы берем ключ ТОЛЬКО из безопасного хранилища Render
raw_key = os.getenv("GOOGLE_API_KEY")

if not raw_key:
    print("!!! ОШИБКА: Ключ не найден в Environment Variables", flush=True)
else:
    # Очистка ключа от пробелов
    CLEAN_KEY = "".join(c for c in raw_key if c.isalnum() or c in "-_")
    genai.configure(api_key=CLEAN_KEY)
    print(f"--> Ключ загружен из настроек. Длина: {len(CLEAN_KEY)}", flush=True)

# 2. Выбираем модель, которая точно работает (из диагностики)
MODEL_NAME = 'gemini-2.0-flash'
model = genai.GenerativeModel(MODEL_NAME)

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

@app.post("/api/analyze")
async def analyze_blood(file: UploadFile = File(...)):
    print(f"\n--> [АНАЛИЗ] Получен файл: {file.filename}", flush=True)
    try:
        content = await file.read()
        
        prompt = """
        Ты врач-гематолог. Проанализируй анализ крови.
        1. Выпиши отклонения.
        2. Объясни их значение.
        3. Дай рекомендации.
        Ответь в формате HTML (теги <b>, <ul>), без ```html.
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
        # Если ошибка 403 - значит ключ снова заблокирован
        if "403" in err:
             return JSONResponse(content={"analysis": "Ошибка 403: Ключ заблокирован Google. Создайте новый и добавьте в Environment."}, status_code=500)
        return JSONResponse(content={"analysis": f"Ошибка сервера: {err}"}, status_code=500)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)

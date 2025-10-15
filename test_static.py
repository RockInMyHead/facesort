"""
Тест статических файлов для Windows
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os

app = FastAPI()

# Монтируем статические файлы
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Главная страница."""
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except Exception as e:
        return HTMLResponse(content=f"<h1>Ошибка загрузки: {e}</h1>")

@app.get("/test")
async def test():
    """Тест статических файлов."""
    return {
        "static_dir_exists": os.path.exists("static"),
        "index_exists": os.path.exists("static/index.html"),
        "app_js_exists": os.path.exists("static/app.js"),
        "current_dir": os.getcwd(),
        "files_in_static": os.listdir("static") if os.path.exists("static") else "static не найден"
    }

if __name__ == "__main__":
    print("🚀 Тест статических файлов...")
    print(f"📁 Текущая директория: {os.getcwd()}")
    print(f"📁 static существует: {os.path.exists('static')}")
    if os.path.exists("static"):
        print(f"📁 Файлы в static: {os.listdir('static')}")
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

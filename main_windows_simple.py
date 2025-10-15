"""
Простая версия для Windows без проблемных зависимостей
"""
import os
import asyncio
import uuid
import time
import tempfile
import re
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional
import unicodedata

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import cv2
import numpy as np
from PIL import Image

# Импортируем только простую версию
from cluster_simple_windows import build_plan_simple, distribute_to_folders, process_group_folder, IMG_EXTS

app = FastAPI(title="Кластеризация лиц", description="API для кластеризации лиц и распределения по группам")

# Монтируем статические файлы
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS middleware для поддержки фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальное состояние приложения
app_state = {
    "queue": [],
    "current_tasks": {},
    "results": {}
}

class QueueItem(BaseModel):
    path: str
    includeExcluded: bool = False

class TaskStatus(BaseModel):
    task_id: str
    status: str
    progress: int
    message: str
    result: Optional[Dict] = None
    error: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
async def root():
    """Главная страница."""
    return HTMLResponse(content=open("static/index.html", "r", encoding="utf-8").read())

@app.get("/api/drives")
async def get_drives():
    """Получить список дисков для Windows."""
    drives = []
    # Для Windows
    for i in range(ord('C'), ord('Z') + 1):
        drive = f"{chr(i)}:\\"
        if os.path.exists(drive):
            drives.append({"name": drive, "path": drive})
    
    # Добавляем Рабочий стол и Документы
    desktop_path = Path(os.path.join(os.path.expanduser("~"), "Desktop"))
    documents_path = Path(os.path.join(os.path.expanduser("~"), "Documents"))
    
    if desktop_path.exists():
        drives.insert(0, {"name": "Рабочий стол", "path": str(desktop_path)})
    if documents_path.exists():
        drives.insert(0, {"name": "Документы", "path": str(documents_path)})

    return {"drives": drives}

@app.get("/api/folder")
async def get_folder_contents(path: str):
    """Возвращает содержимое папки (подпапки и изображения)."""
    try:
        decoded_path = path.replace('%20', ' ')
        folder_path = Path(decoded_path)

        if not folder_path.exists() or not folder_path.is_dir():
            raise HTTPException(status_code=404, detail="Папка не найдена")

        folders = []
        images = []
        for item in folder_path.iterdir():
            if item.is_dir():
                folders.append({"name": item.name, "path": str(item)})
            elif item.is_file() and item.suffix.lower() in IMG_EXTS:
                images.append({"name": item.name, "path": str(item)})
        
        # Объединяем для фронтенда
        contents = []
        for f in folders:
            contents.append({"name": f["name"], "path": f["path"], "is_directory": True})
        for i in images:
            contents.append({"name": i["name"], "path": i["path"], "is_directory": False})

        return {
            "path": str(folder_path),
            "folders": folders,
            "images": images,
            "contents": contents
        }
    except Exception as e:
        print(f"❌ Ошибка доступа к папке {path}: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка доступа к папке: {e}")

@app.get("/api/image/preview")
async def get_image_preview(path: str, size: int = 150):
    """Получить превью изображения."""
    try:
        decoded_path = path.replace('%20', ' ')
        image_path = Path(decoded_path)
        
        if not image_path.exists() or not image_path.is_file():
            raise HTTPException(status_code=404, detail="Изображение не найдено")
        
        if image_path.suffix.lower() not in IMG_EXTS:
            raise HTTPException(status_code=400, detail="Не является изображением")
        
        img = cv2.imread(str(image_path))
        if img is None:
            raise HTTPException(status_code=400, detail="Не удалось загрузить изображение")
        
        height, width = img.shape[:2]
        if width > height:
            new_width = size
            new_height = int(height * size / width)
        else:
            new_height = size
            new_width = int(width * size / height)
        
        resized = cv2.resize(img, (new_width, new_height))
        
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        
        img_bytes = BytesIO()
        pil_img.save(img_bytes, format='JPEG', quality=85)
        img_bytes.seek(0)
        
        return Response(
            content=img_bytes.getvalue(),
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=3600"}
        )
        
    except Exception as e:
        print(f"❌ Ошибка превью для {path}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/queue/add")
async def add_to_queue(request: QueueItem):
    """Добавляет папку в очередь на обработку."""
    try:
        folder_path = Path(request.path)
        if not folder_path.exists() or not folder_path.is_dir():
            raise HTTPException(status_code=404, detail="Папка не найдена")
        
        if request.path not in app_state["queue"]:
            app_state["queue"].append(request.path)
            return {"message": f"Папка добавлена в очередь: {request.path}"}
        return {"message": f"Папка уже в очереди: {request.path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/queue")
async def get_queue():
    """Возвращает текущую очередь обработки."""
    return {"queue": app_state["queue"]}

@app.get("/api/tasks", response_model=List[TaskStatus])
async def get_tasks():
    """Возвращает статус всех текущих и завершенных задач."""
    return list(app_state["current_tasks"].values())

async def process_folder_task(task_id: str, folder_path: str, include_excluded: bool):
    """Фоновая задача для обработки папки."""
    app_state["current_tasks"][task_id] = {
        "task_id": task_id,
        "status": "running",
        "progress": 0,
        "message": "Начинаем обработку...",
        "result": None
    }

    def progress_callback(progress: float, message: str):
        if isinstance(progress, (int, float)):
            app_state["current_tasks"][task_id]["progress"] = int(progress)
        app_state["current_tasks"][task_id]["message"] = message
        print(f"📊 [TASK] {message}")

    try:
        print(f"🚀 [TASK] Начинаем обработку: {folder_path}")

        # Нормализация пути
        normalized_path = unicodedata.normalize('NFKC', folder_path)
        normalized_path = normalized_path.replace('\u00A0', ' ').replace('\xa0', ' ')
        
        path = Path(normalized_path)

        if not path.exists():
            raise FileNotFoundError(f"Папка не найдена: {folder_path}")

        # Шаг 1: Построение плана кластеризации
        progress_callback(0, "🔍 Поиск лиц и извлечение эмбеддингов...")
        
        # Используем build_plan_simple
        plan = build_plan_simple(path, n_clusters=8, progress_callback=progress_callback)

        if not isinstance(plan, dict) or not plan.get("cluster_map"):
            app_state["current_tasks"][task_id]["status"] = "completed"
            app_state["current_tasks"][task_id]["progress"] = 100
            app_state["current_tasks"][task_id]["message"] = "Нет данных для обработки или кластеров не создано."
            print(f"⚠️ [TASK] Нет данных для обработки или кластеров не создано для {folder_path}")
            return

        progress_callback(80, f"✅ Готово! {len(plan['cluster_map'])} кластеров")

        # Шаг 2: Распределение файлов по папкам
        progress_callback(85, "📁 Распределение файлов по папкам...")
        moved_count, copied_count, total_clusters = distribute_to_folders(plan, path, 1, progress_callback)

        app_state["current_tasks"][task_id]["status"] = "completed"
        app_state["current_tasks"][task_id]["progress"] = 100
        app_state["current_tasks"][task_id]["message"] = f"Готово! Перемещено: {moved_count}, скопировано: {copied_count}"
        app_state["current_tasks"][task_id]["result"] = {
            "moved": moved_count,
            "copied": copied_count,
            "clusters": total_clusters
        }
        print(f"✅ [TASK] Завершено: {folder_path}")

    except Exception as e:
        print(f"❌ [TASK] Ошибка при обработке {folder_path}: {e}")
        app_state["current_tasks"][task_id]["status"] = "error"
        app_state["current_tasks"][task_id]["error"] = str(e)
        app_state["current_tasks"][task_id]["message"] = f"Ошибка: {e}"

    finally:
        # Удаляем из очереди после завершения (успех или ошибка)
        if folder_path in app_state["queue"]:
            app_state["queue"].remove(folder_path)

@app.post("/api/process")
async def process_queue(background_tasks: BackgroundTasks, includeExcluded: bool = False):
    """Запускает обработку всех папок в очереди."""
    if not app_state["queue"]:
        raise HTTPException(status_code=400, detail="Очередь пуста.")

    task_ids = []
    for folder_path in app_state["queue"]:
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)
        background_tasks.add_task(process_folder_task, task_id, folder_path, includeExcluded)
    
    return {"message": "Обработка запущена", "task_ids": task_ids}

if __name__ == "__main__":
    print("🚀 Запуск упрощенного сервера для Windows...")
    print("📍 Открыть: http://localhost:8000")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

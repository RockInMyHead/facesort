"""
Упрощенная версия main.py без проблемных зависимостей.
"""
import os
import asyncio
import concurrent.futures
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
from cluster_simple import build_plan_simple as build_plan_advanced, distribute_to_folders, process_group_folder, IMG_EXTS

app = FastAPI(title="Кластеризация лиц", description="API для кластеризации лиц и распределения по группам")
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

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

class TaskResult(BaseModel):
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
    """Получить список дисков."""
    import platform
    drives = []
    
    if platform.system() == "Darwin":  # macOS
        drives = [
            {"name": "🏠 Домашняя папка", "path": str(Path.home())},
            {"name": "📁 Рабочий стол", "path": str(Path.home() / "Desktop")},
            {"name": "📷 Изображения", "path": str(Path.home() / "Pictures")},
            {"name": "📁 Документы", "path": str(Path.home() / "Documents")},
            {"name": "💾 Загрузки", "path": str(Path.home() / "Downloads")}
        ]
    elif platform.system() == "Windows":
        import string
        drives = [{"name": f"💾 Диск {d}:", "path": f"{d}:\\"} for d in string.ascii_uppercase if Path(f"{d}:\\").exists()]
    else:  # Linux
        drives = [
            {"name": "🏠 Домашняя папка", "path": str(Path.home())},
            {"name": "📁 Рабочий стол", "path": str(Path.home() / "Desktop")},
            {"name": "📷 Изображения", "path": str(Path.home() / "Pictures")},
            {"name": "📁 Документы", "path": str(Path.home() / "Documents")},
            {"name": "💾 Загрузки", "path": str(Path.home() / "Downloads")}
        ]
    
    return {"drives": drives}

@app.get("/api/folder")
async def get_folder_contents(path: str):
    """Получить содержимое папки."""
    try:
        folder_path = Path(path)
        if not folder_path.exists() or not folder_path.is_dir():
            raise HTTPException(status_code=404, detail="Папка не найдена")
        
        # Получаем список папок и изображений
        folders = []
        images = []
        
        for item in sorted(folder_path.iterdir()):
            if item.is_dir():
                folders.append({
                    "name": item.name,
                    "path": str(item),
                    "type": "folder"
                })
            elif item.is_file() and item.suffix.lower() in IMG_EXTS:
                images.append({
                    "name": item.name,
                    "path": str(item),
                    "type": "image"
                })
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
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/queue")
async def get_queue():
    """Получить очередь."""
    return {"queue": app_state["queue"]}

@app.post("/api/queue/add")
async def add_to_queue(item: QueueItem):
    """Добавить папку в очередь."""
    # Нормализация пути
    normalized_path = unicodedata.normalize('NFKC', item.path)
    # Заменяем неразрывные пробелы и другие проблемные символы
    normalized_path = normalized_path.replace('\u00A0', ' ').replace('\xa0', ' ')
    
    if normalized_path not in app_state["queue"]:
        app_state["queue"].append(normalized_path)
        return {"message": f"Папка добавлена в очередь: {normalized_path}"}
    else:
        return {"message": f"Папка уже в очереди: {normalized_path}"}

@app.delete("/api/queue/clear")
async def clear_queue():
    """Очистить очередь."""
    app_state["queue"] = []
    return {"message": "Очередь очищена"}

@app.post("/api/process")
async def process_queue(includeExcluded: bool = False, background_tasks: BackgroundTasks = None):
    """Обработать очередь."""
    if not app_state["queue"]:
        raise HTTPException(status_code=400, detail="Очередь пуста")
    
    task_ids = []
    for folder_path in app_state["queue"]:
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)
        
        app_state["current_tasks"][task_id] = {
            "status": "running",
            "progress": 0,
            "message": "Обработка запущена",
            "folder_path": folder_path,
            "include_excluded": includeExcluded
        }
        
        # Запускаем обработку в фоне
        background_tasks.add_task(process_folder_task, task_id, folder_path, includeExcluded)
    
    # Очищаем очередь
    app_state["queue"] = []
    
    return {"message": "Обработка запущена", "task_ids": task_ids}

async def process_folder_task(task_id: str, folder_path: str, include_excluded: bool):
    """Обрабатывает папку."""
    try:
        print(f"🚀 [TASK] Начинаем обработку: {folder_path}")
        
        # Нормализация пути
        normalized_path = unicodedata.normalize('NFKC', folder_path)
        # Заменяем неразрывные пробелы и другие проблемные символы
        normalized_path = normalized_path.replace('\u00A0', ' ').replace('\xa0', ' ')
        
        # Если путь все еще не существует, пробуем найти папку с похожим именем
        if not Path(normalized_path).exists():
            parent_dir = Path(normalized_path).parent
            if parent_dir.exists():
                for item in parent_dir.iterdir():
                    if item.is_dir() and '116_Даша' in item.name:
                        print(f"🔍 Найдена папка: {item.name}")
                        normalized_path = str(item)
                        break
        path = Path(normalized_path)
        
        if not path.exists():
            app_state["current_tasks"][task_id]["status"] = "error"
            app_state["current_tasks"][task_id]["error"] = f"Путь не существует: {folder_path}"
            return
        
        def progress_callback(message: str, progress: int):
            if isinstance(progress, (int, float)):
                app_state["current_tasks"][task_id]["progress"] = min(100, max(0, int(progress)))
            app_state["current_tasks"][task_id]["message"] = message
            print(f"📊 [TASK] {message}")
        
        # Запускаем кластеризацию
        loop = asyncio.get_event_loop()
        plan = await loop.run_in_executor(
            executor,
            build_plan_advanced,
            path,
            3,  # n_clusters
            progress_callback
        )
        
        # Проверка результата
        if not isinstance(plan, dict):
            app_state["current_tasks"][task_id]["status"] = "completed"
            app_state["current_tasks"][task_id]["progress"] = 100
            app_state["current_tasks"][task_id]["message"] = "Нет данных для обработки"
            return
        
        # Распределение по папкам
        progress_callback("📁 Распределение файлов по папкам...", 90)
        
        moved, copied, final_cluster = await loop.run_in_executor(
            executor,
            distribute_to_folders,
            plan,
            path,
            1,
            progress_callback
        )
        
        # Завершение
        app_state["current_tasks"][task_id]["status"] = "completed"
        app_state["current_tasks"][task_id]["progress"] = 100
        app_state["current_tasks"][task_id]["message"] = f"Готово! Перемещено: {moved}, скопировано: {copied}"
        app_state["current_tasks"][task_id]["result"] = {
            "moved": moved,
            "copied": copied,
            "clusters": final_cluster - 1
        }
        
        print(f"✅ [TASK] Завершено: {folder_path}")
        
    except Exception as e:
        print(f"❌ [TASK] Ошибка: {e}")
        app_state["current_tasks"][task_id]["status"] = "error"
        app_state["current_tasks"][task_id]["error"] = str(e)

@app.get("/api/tasks")
async def get_tasks():
    """Получить список задач."""
    return {"tasks": list(app_state["current_tasks"].values())}

@app.delete("/api/tasks/clear")
async def clear_completed_tasks():
    """Очистить завершенные задачи."""
    completed_tasks = [tid for tid, task in app_state["current_tasks"].items() 
                      if task["status"] in ["completed", "error"]]
    for tid in completed_tasks:
        del app_state["current_tasks"][tid]
    return {"message": f"Удалено {len(completed_tasks)} задач"}

@app.get("/api/image/preview")
async def get_image_preview(path: str, size: int = 150):
    """Получить превью изображения."""
    try:
        # URL декодирование пути
        import urllib.parse
        decoded_path = urllib.parse.unquote(path)
        
        image_path = Path(decoded_path)
        if not image_path.exists() or not image_path.is_file():
            raise HTTPException(status_code=404, detail="Изображение не найдено")
        
        # Проверяем, что это изображение
        if image_path.suffix.lower() not in IMG_EXTS:
            raise HTTPException(status_code=400, detail="Не является изображением")
        
        # Читаем изображение
        img = cv2.imread(str(image_path))
        if img is None:
            raise HTTPException(status_code=400, detail="Не удалось загрузить изображение")
        
        # Изменяем размер
        height, width = img.shape[:2]
        if width > height:
            new_width = size
            new_height = int(height * size / width)
        else:
            new_height = size
            new_width = int(width * size / height)
        
        resized = cv2.resize(img, (new_width, new_height))
        
        # Конвертируем в RGB и затем в JPEG
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        
        # Сохраняем в байты
        img_bytes = BytesIO()
        pil_img.save(img_bytes, format='JPEG', quality=85)
        img_bytes.seek(0)
        
        return Response(
            content=img_bytes.getvalue(),
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=3600"}
        )
        
    except Exception as e:
        print(f"❌ Ошибка превью: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Статические файлы
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Запуск упрощенного сервера кластеризации...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

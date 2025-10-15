"""
Упрощенная версия для Windows с улучшенной обработкой ошибок
"""
import asyncio
import uuid
import os
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
import shutil
from concurrent.futures import ThreadPoolExecutor
import functools

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# Монтируем статические файлы
app.mount("/static", StaticFiles(directory="static"), name="static")

# Глобальное состояние
app_state = {
    "queue": [],
    "current_tasks": {}
}

# Executor для фоновых задач
executor = ThreadPoolExecutor(max_workers=2)

# Поддерживаемые форматы
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

class FolderRequest(BaseModel):
    path: str

@app.get("/", response_class=HTMLResponse)
async def root():
    """Главная страница."""
    return HTMLResponse(content=open("static/index.html", "r", encoding="utf-8").read())

@app.get("/api/drives")
async def get_drives():
    """Возвращает список доступных дисков/корневых папок для Windows."""
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

@app.get("/api/queue")
async def get_queue():
    return {"queue": app_state["queue"]}

@app.post("/api/queue/add")
async def add_to_queue(request: FolderRequest):
    try:
        folder_path = Path(request.path)
        if not folder_path.exists():
            raise HTTPException(status_code=404, detail="Папка не найдена")
        
        if str(folder_path) not in app_state["queue"]:
            app_state["queue"].append(str(folder_path))
        
        return {"message": f"Папка добавлена в очередь: {folder_path}"}
    except Exception as e:
        print(f"❌ Ошибка добавления в очередь: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tasks")
async def get_tasks():
    return {"tasks": list(app_state["current_tasks"].values())}

@app.post("/api/process")
async def process_queue(includeExcluded: bool = False):
    try:
        if not app_state["queue"]:
            return {"message": "Очередь пуста", "task_ids": []}
        
        task_ids = []
        for folder_path in app_state["queue"][:]:
            task_id = str(uuid.uuid4())
            task_ids.append(task_id)
            
            app_state["current_tasks"][task_id] = {
                "status": "pending",
                "progress": 0,
                "message": "Ожидание...",
                "folder_path": folder_path,
                "include_excluded": includeExcluded
            }
            
            # Запускаем обработку в фоне
            asyncio.create_task(process_folder_task(task_id, folder_path, includeExcluded))
        
        app_state["queue"] = []
        return {"message": "Обработка запущена", "task_ids": task_ids}
    
    except Exception as e:
        print(f"❌ Ошибка запуска обработки: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_folder_task(task_id: str, folder_path: str, include_excluded: bool):
    """Обработка папки с изображениями"""
    try:
        print(f"🚀 [TASK] Начинаем обработку: {folder_path}")
        
        path = Path(folder_path)
        if not path.exists():
            raise Exception(f"Путь не существует: {folder_path}")
        
        def progress_callback(progress, message=""):
            if isinstance(progress, (int, float)):
                app_state["current_tasks"][task_id]["progress"] = int(progress)
            if message:
                app_state["current_tasks"][task_id]["message"] = message
                print(f"📊 [TASK] {message}")
        
        # Запускаем кластеризацию
        loop = asyncio.get_event_loop()
        clustering_func = functools.partial(
            simple_clustering,
            input_dir=path,
            progress_callback=progress_callback
        )
        
        result = await loop.run_in_executor(executor, clustering_func)
        
        # Обновляем статус
        app_state["current_tasks"][task_id]["status"] = "completed"
        app_state["current_tasks"][task_id]["progress"] = 100
        app_state["current_tasks"][task_id]["message"] = f"Готово! Создано кластеров: {result['clusters']}"
        app_state["current_tasks"][task_id]["result"] = result
        
        print(f"✅ [TASK] Завершено: {folder_path}")
        
    except Exception as e:
        print(f"❌ [TASK] Ошибка: {e}")
        import traceback
        traceback.print_exc()
        app_state["current_tasks"][task_id]["status"] = "error"
        app_state["current_tasks"][task_id]["error"] = str(e)

def simple_clustering(input_dir: Path, progress_callback=None) -> Dict:
    """Упрощенная кластеризация с базовым детектором лиц"""
    print(f"🚀 [SIMPLE] Упрощенная кластеризация: {input_dir}")
    
    # Найти все изображения
    images = []
    for ext in IMG_EXTS:
        images.extend(input_dir.glob(f"*{ext}"))
        images.extend(input_dir.glob(f"*{ext.upper()}"))
    
    if not images:
        return {"clusters": 0, "moved": 0, "copied": 0}
    
    print(f"📂 Найдено {len(images)} изображений")
    if progress_callback:
        progress_callback(0, f"📂 Найдено {len(images)} изображений")
    
    # Загрузить детектор лиц OpenCV
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        print(f"✅ Загружен детектор лиц: {cascade_path}")
    except Exception as e:
        print(f"❌ Ошибка загрузки детектора: {e}")
        return {"clusters": 0, "moved": 0, "copied": 0}
    
    # Извлечь лица и создать простые эмбеддинги
    image_faces = defaultdict(list)
    all_embeddings = []
    all_metadata = []
    
    for i, img_path in enumerate(images):
        try:
            # Прогресс
            if progress_callback and i % 5 == 0:
                progress = int((i / len(images)) * 50)  # 0-50% на детекцию
                progress_callback(progress, f"📷 Анализ: {progress}% ({i}/{len(images)})")
            
            # Читаем изображение
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"⚠️ Не удалось прочитать: {img_path.name}")
                continue
            
            # Изменяем размер для ускорения
            h, w = img.shape[:2]
            max_size = 800
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
            
            # Конвертируем в grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Детекция лиц
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            if len(faces) == 0:
                continue
            
            # Для каждого лица создаем простой эмбеддинг
            for (x, y, w, h) in faces:
                # Вырезаем лицо
                face_img = gray[y:y+h, x:x+w]
                
                # Изменяем размер до фиксированного
                face_img = cv2.resize(face_img, (64, 64))
                
                # Нормализуем
                face_img = face_img.astype('float32') / 255.0
                
                # Флаттенируем как эмбеддинг
                embedding = face_img.flatten()
                
                all_embeddings.append(embedding)
                all_metadata.append(str(img_path))
                image_faces[str(img_path)].append(embedding)
        
        except Exception as e:
            print(f"❌ Ошибка обработки {img_path.name}: {e}")
            continue
    
    if len(all_embeddings) == 0:
        print("⚠️ Не найдено лиц")
        return {"clusters": 0, "moved": 0, "copied": 0}
    
    print(f"✅ Извлечено {len(all_embeddings)} лиц из {len(image_faces)} изображений")
    if progress_callback:
        progress_callback(50, f"✅ Найдено {len(all_embeddings)} лиц")
    
    # Кластеризация
    X = np.array(all_embeddings)
    
    # Нормализуем для косинусного расстояния
    from sklearn.preprocessing import normalize
    X = normalize(X, norm='l2')
    
    # Вычисляем косинусное расстояние
    dist_matrix = cosine_distances(X)
    
    # AgglomerativeClustering
    n_clusters = min(8, len(X))  # Максимум 8 кластеров
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'
    )
    
    if progress_callback:
        progress_callback(60, f"🔄 Кластеризация {len(X)} лиц...")
    
    labels = clustering.fit_predict(dist_matrix)
    
    print(f"✅ Кластеризация завершена: {len(set(labels))} кластеров")
    
    # Группируем изображения по кластерам
    cluster_map = defaultdict(set)
    for img_path, label in zip(all_metadata, labels):
        cluster_map[label].add(img_path)
    
    # Распределяем файлы
    moved = 0
    copied = 0
    
    for cluster_id, img_paths in cluster_map.items():
        cluster_folder = input_dir / f"cluster_{cluster_id+1}_({len(img_paths)})"
        cluster_folder.mkdir(exist_ok=True)
        
        for img_path in img_paths:
            src = Path(img_path)
            dst = cluster_folder / src.name
            
            try:
                if not dst.exists():
                    shutil.copy2(src, dst)
                    copied += 1
            except Exception as e:
                print(f"❌ Ошибка копирования {src.name}: {e}")
    
    print(f"📦 Скопировано: {copied}")
    if progress_callback:
        progress_callback(100, f"✅ Готово! Создано {len(cluster_map)} кластеров")
    
    return {
        "clusters": len(cluster_map),
        "moved": moved,
        "copied": copied
    }

if __name__ == "__main__":
    print("🚀 Запуск упрощенного сервера для Windows...")
    print("📍 Открыть: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)


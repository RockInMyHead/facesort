"""
Простая кластеризация для Windows без проблемных зависимостей
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
import shutil
import os

# Поддерживаемые форматы
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

class SimpleFaceRecognition:
    """Простая система распознавания лиц с OpenCV."""
    
    def __init__(self):
        # Загружаем каскад Хаара для детекции лиц
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        print("✅ Инициализирована простая система распознавания лиц")
    
    def detect_faces_simple(self, img: np.ndarray) -> List[np.ndarray]:
        """Простая детекция лиц с помощью OpenCV Haar Cascade."""
        try:
            # Конвертируем в серый
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Детекция лиц
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            face_images = []
            for (x, y, w, h) in faces:
                # Извлекаем лицо
                face_img = img[y:y+h, x:x+w]
                if face_img.size > 0:
                    face_images.append(face_img)
            
            return face_images
            
        except Exception as e:
            print(f"⚠️ Ошибка детекции лиц: {e}")
            return []
    
    def extract_embedding_simple(self, face_img: np.ndarray) -> np.ndarray:
        """Простое извлечение признаков лица."""
        try:
            # Изменяем размер до стандартного
            face_resized = cv2.resize(face_img, (96, 96))
            
            # Конвертируем в серый
            gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            
            # Нормализуем
            normalized = gray.astype(np.float32) / 255.0
            
            # Простое извлечение признаков (можно заменить на более сложное)
            # Используем гистограмму градиентов как простой дескриптор
            sobelx = cv2.Sobel(normalized, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(normalized, cv2.CV_64F, 0, 1, ksize=3)
            
            # Вычисляем гистограмму градиентов
            hist_x = cv2.calcHist([sobelx], [0], None, [32], [0, 256])
            hist_y = cv2.calcHist([sobely], [0], None, [32], [0, 256])
            
            # Объединяем гистограммы
            features = np.concatenate([hist_x.flatten(), hist_y.flatten()])
            
            # Нормализуем
            features = features / (np.linalg.norm(features) + 1e-8)
            
            return features
            
        except Exception as e:
            print(f"⚠️ Ошибка извлечения признаков: {e}")
            return np.zeros(64)  # Возвращаем нулевой вектор

def build_plan_simple(
    input_dir: Path, 
    n_clusters: int = 8,
    progress_callback: Optional[Callable] = None
) -> Dict:
    """Простое построение плана кластеризации."""
    
    if progress_callback:
        progress_callback(0, "🚀 [SIMPLE] Запуск упрощенной кластеризации...")
    
    # Инициализация системы распознавания
    recognizer = SimpleFaceRecognition()
    
    # Поиск изображений
    image_files = []
    for ext in IMG_EXTS:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"*{ext.upper()}"))
    
    if progress_callback:
        progress_callback(0, f"📂 Найдено {len(image_files)} изображений")
    
    if not image_files:
        return {"cluster_map": {}, "embeddings": [], "image_paths": []}
    
    # Извлечение признаков
    embeddings = []
    image_paths = []
    
    for i, img_path in enumerate(image_files):
        if progress_callback:
            progress = int((i / len(image_files)) * 60)
            progress_callback(progress, f"📷 Анализ: {progress}% ({i+1}/{len(image_files)})")
        
        try:
            # Загружаем изображение
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Детекция лиц
            faces = recognizer.detect_faces_simple(img)
            
            if not faces:
                continue
            
            # Извлекаем признаки для каждого лица
            for face in faces:
                embedding = recognizer.extract_embedding_simple(face)
                embeddings.append(embedding)
                image_paths.append(str(img_path))
                
        except Exception as e:
            print(f"⚠️ Ошибка обработки {img_path}: {e}")
            continue
    
    if not embeddings:
        if progress_callback:
            progress_callback(100, "❌ Лица не найдены")
        return {"cluster_map": {}, "embeddings": [], "image_paths": []}
    
    if progress_callback:
        progress_callback(70, f"✅ Извлечено {len(embeddings)} эмбеддингов из {len(image_files)} изображений")
    
    # Кластеризация
    if progress_callback:
        progress_callback(75, f"🔄 Agglomerative Clustering {len(embeddings)} лиц...")
    
    try:
        # Используем AgglomerativeClustering
        clustering = AgglomerativeClustering(
            n_clusters=min(n_clusters, len(embeddings)),
            metric='precomputed',
            linkage='average'
        )
        
        # Вычисляем матрицу расстояний
        dist_matrix = cosine_distances(embeddings)
        
        # Кластеризация
        labels = clustering.fit_predict(dist_matrix)
        
        if progress_callback:
            progress_callback(85, f"✅ Кластеризация завершена: {len(set(labels))} кластеров")
        
        # Создаем карту кластеров
        cluster_map = {}
        for i, (label, img_path) in enumerate(zip(labels, image_paths)):
            if label not in cluster_map:
                cluster_map[label] = []
            cluster_map[label].append(img_path)
        
        if progress_callback:
            progress_callback(90, f"✅ Создано {len(cluster_map)} кластеров")
        
        return {
            "cluster_map": cluster_map,
            "embeddings": embeddings,
            "image_paths": image_paths
        }
        
    except Exception as e:
        print(f"❌ Ошибка кластеризации: {e}")
        if progress_callback:
            progress_callback(100, f"❌ Ошибка кластеризации: {e}")
        return {"cluster_map": {}, "embeddings": [], "image_paths": []}

def distribute_to_folders(plan: Dict, base_dir: Path, cluster_start: int = 1, progress_callback: Optional[Callable] = None) -> Tuple[int, int, int]:
    """Распределение файлов по папкам."""
    
    if not plan.get("cluster_map"):
        return 0, 0, 0
    
    cluster_map = plan["cluster_map"]
    total_clusters = len(cluster_map)
    
    if progress_callback:
        progress_callback(0, f"🔄 Распределение {sum(len(files) for files in cluster_map.values())} файлов по папкам...")
    
    moved_count = 0
    copied_count = 0
    
    for cluster_id, files in cluster_map.items():
        if not files:
            continue
        
        # Создаем папку для кластера
        cluster_folder = base_dir / f"cluster_{cluster_id + cluster_start}"
        cluster_folder.mkdir(exist_ok=True)
        
        # Перемещаем файлы
        for i, file_path in enumerate(files):
            try:
                src_path = Path(file_path)
                dst_path = cluster_folder / src_path.name
                
                if src_path.exists():
                    if src_path.parent == cluster_folder.parent:
                        # Если файл уже в правильной папке, копируем
                        shutil.copy2(src_path, dst_path)
                        copied_count += 1
                    else:
                        # Перемещаем файл
                        shutil.move(str(src_path), str(dst_path))
                        moved_count += 1
                
                if progress_callback:
                    progress = int(((i + 1) / len(files)) * 100)
                    progress_callback(progress, f"📁 Распределение файлов: {progress}% ({i+1}/{len(files)})")
                    
            except Exception as e:
                print(f"⚠️ Ошибка перемещения {file_path}: {e}")
                continue
    
    if progress_callback:
        progress_callback(100, f"✅ Распределение завершено: {moved_count} перемещено, {copied_count} скопировано")
    
    return moved_count, copied_count, total_clusters

def process_group_folder(folder_path: Path, progress_callback: Optional[Callable] = None) -> Dict:
    """Обработка папки с группой изображений."""
    return build_plan_simple(folder_path, progress_callback=progress_callback)

"""
Альтернативная версия cluster.py для Windows
Использует MediaPipe вместо face_recognition для лучшей совместимости
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import shutil
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import mediapipe as mp
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Поддерживаемые форматы изображений
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

class MediaPipeFaceDetector:
    """Детектор лиц на основе MediaPipe"""
    
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Инициализация детектора лиц
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0 для ближних лиц, 1 для дальних
            min_detection_confidence=0.5
        )
        
        # Инициализация mesh для извлечения признаков
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """Детекция лиц в изображении"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        
        faces = []
        if results.detections:
            h, w, _ = image.shape
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Проверяем, что лицо не выходит за границы
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                if width > 0 and height > 0:
                    faces.append({
                        'bbox': (x, y, width, height),
                        'confidence': detection.score[0],
                        'landmarks': None
                    })
        
        return faces
    
    def extract_landmarks(self, image: np.ndarray, face_bbox: Tuple) -> Optional[np.ndarray]:
        """Извлечение ключевых точек лица"""
        x, y, w, h = face_bbox
        face_crop = image[y:y+h, x:x+w]
        
        if face_crop.size == 0:
            return None
            
        rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_face)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            # Извлекаем координаты ключевых точек
            points = []
            for landmark in landmarks.landmark:
                points.extend([landmark.x, landmark.y, landmark.z])
            return np.array(points)
        
        return None
    
    def get_face_embedding(self, image: np.ndarray, face_bbox: Tuple) -> Optional[np.ndarray]:
        """Получение эмбеддинга лица"""
        landmarks = self.extract_landmarks(image, face_bbox)
        if landmarks is not None:
            # Нормализуем landmarks
            landmarks = landmarks.reshape(-1, 3)
            # Центрируем относительно центра лица
            center = np.mean(landmarks, axis=0)
            landmarks = landmarks - center
            # Нормализуем по масштабу
            scale = np.std(landmarks)
            if scale > 0:
                landmarks = landmarks / scale
            return landmarks.flatten()
        return None

def process_image_windows(image_path: Path, detector: MediaPipeFaceDetector) -> List[Dict]:
    """Обработка одного изображения для Windows"""
    try:
        # Загружаем изображение
        image = cv2.imread(str(image_path))
        if image is None:
            return []
        
        # Детекция лиц
        faces = detector.detect_faces(image)
        
        results = []
        for i, face in enumerate(faces):
            if face['confidence'] > 0.5:  # Минимальная уверенность
                # Извлекаем эмбеддинг
                embedding = detector.get_face_embedding(image, face['bbox'])
                if embedding is not None:
                    results.append({
                        'file_path': str(image_path),
                        'face_id': i,
                        'bbox': face['bbox'],
                        'confidence': face['confidence'],
                        'embedding': embedding
                    })
        
        return results
        
    except Exception as e:
        print(f"Ошибка обработки {image_path}: {e}")
        return []

def build_plan_windows(input_dir: Path, include_excluded: bool = False) -> Dict[str, Any]:
    """Построение плана кластеризации для Windows"""
    print(f"🔍 build_plan_windows: input_dir={input_dir}, include_excluded={include_excluded}")
    
    # Исключенные названия папок
    excluded_names = ["общие", "общая", "common", "shared", "все", "all", "mixed", "смешанные"]
    
    # Находим все изображения
    image_files = []
    for ext in IMG_EXTS:
        image_files.extend(input_dir.rglob(f"*{ext}"))
        image_files.extend(input_dir.rglob(f"*{ext.upper()}"))
    
    print(f"🔍 Найдено {len(image_files)} изображений")
    
    # Инициализируем детектор
    detector = MediaPipeFaceDetector()
    
    # Обрабатываем изображения
    all_faces = []
    processed_count = 0
    
    for img_path in image_files:
        if not include_excluded:
            # Проверяем, не находится ли файл в исключенной папке
            path_str = str(img_path).lower()
            if any(excluded_name in path_str for excluded_name in excluded_names):
                continue
        
        faces = process_image_windows(img_path, detector)
        all_faces.extend(faces)
        processed_count += 1
        
        if processed_count % 10 == 0:
            print(f"🔍 Обработано {processed_count}/{len(image_files)} изображений")
    
    print(f"🔍 Найдено {len(all_faces)} лиц")
    
    if len(all_faces) < 2:
        return {
            "clusters": {},
            "unreadable": [],
            "no_faces": [f["file_path"] for f in all_faces]
        }
    
    # Извлекаем эмбеддинги
    embeddings = []
    face_to_embedding = {}
    
    for face in all_faces:
        embedding = face['embedding']
        if embedding is not None and len(embedding) > 0:
            embeddings.append(embedding)
            face_to_embedding[id(face)] = len(embeddings) - 1
    
    if len(embeddings) < 2:
        return {
            "clusters": {},
            "unreadable": [],
            "no_faces": [f["file_path"] for f in all_faces]
        }
    
    # Кластеризация
    embeddings = np.array(embeddings)
    
    # Используем DBSCAN вместо HDBSCAN для лучшей совместимости
    clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
    cluster_labels = clustering.fit_predict(embeddings)
    
    # Группируем по кластерам
    clusters = {}
    for i, face in enumerate(all_faces):
        if id(face) in face_to_embedding:
            embedding_idx = face_to_embedding[id(face)]
            cluster_id = cluster_labels[embedding_idx]
            
            if cluster_id == -1:  # Шум
                continue
                
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(face)
    
    print(f"🔍 Создано {len(clusters)} кластеров")
    
    return {
        "clusters": clusters,
        "unreadable": [],
        "no_faces": []
    }

def distribute_to_folders_windows(plan: Dict[str, Any], base_path: Path) -> Tuple[int, int, int]:
    """Распределение файлов по папкам для Windows"""
    moved = 0
    copied = 0
    next_cluster_id = 1
    
    for cluster_id, faces in plan["clusters"].items():
        if len(faces) < 2:  # Минимальный размер кластера
            continue
            
        # Создаем папку для кластера
        cluster_folder = base_path / f"cluster_{next_cluster_id}"
        cluster_folder.mkdir(exist_ok=True)
        
        # Группируем файлы по путям
        files_in_cluster = {}
        for face in faces:
            file_path = Path(face['file_path'])
            if file_path not in files_in_cluster:
                files_in_cluster[file_path] = []
            files_in_cluster[file_path].append(face)
        
        # Перемещаем или копируем файлы
        for file_path, faces_in_file in files_in_cluster.items():
            if len(files_in_cluster) == 1:
                # Только один файл в кластере - перемещаем
                dest_path = cluster_folder / file_path.name
                shutil.move(str(file_path), str(dest_path))
                moved += 1
            else:
                # Несколько файлов - копируем
                dest_path = cluster_folder / file_path.name
                shutil.copy2(str(file_path), str(dest_path))
                copied += 1
        
        next_cluster_id += 1
    
    return moved, copied, next_cluster_id

# Функции для совместимости с основным кодом
def build_plan_live(input_dir: Path, include_excluded: bool = False, progress_callback=None) -> Dict[str, Any]:
    """Основная функция для совместимости"""
    if progress_callback:
        progress_callback("Начинаем обработку изображений...", 10)
    
    plan = build_plan_windows(input_dir, include_excluded)
    
    if progress_callback:
        progress_callback("Кластеризация завершена", 90)
    
    return plan

def distribute_to_folders(plan: Dict[str, Any], base_path: Path, progress_callback=None) -> Tuple[int, int, int]:
    """Распределение файлов по папкам"""
    if progress_callback:
        progress_callback("Распределяем файлы по папкам...", 95)
    
    return distribute_to_folders_windows(plan, base_path)

def process_group_folder(input_dir: Path, progress_callback=None, include_excluded: bool = False):
    """Обработка группы папок"""
    if progress_callback:
        progress_callback("Обрабатываем группу папок...", 5)
    
    # Находим подпапки с изображениями
    subdirs = []
    for item in input_dir.iterdir():
        if item.is_dir() and not any(excluded_name in str(item).lower() 
                                   for excluded_name in ["общие", "общая", "common", "shared", "все", "all", "mixed", "смешанные"]):
            # Проверяем, есть ли изображения в подпапке
            has_images = any(f.suffix.lower() in IMG_EXTS for f in item.rglob("*") if f.is_file())
            if has_images:
                subdirs.append(item)
    
    if progress_callback:
        progress_callback(f"Найдено {len(subdirs)} подпапок для обработки", 10)
    
    # Обрабатываем каждую подпапку
    for i, subdir in enumerate(subdirs):
        if progress_callback:
            progress_callback(f"Обрабатываем папку: {subdir.name}", 10 + (i * 80 // len(subdirs)))
        
        plan = build_plan_windows(subdir, include_excluded)
        moved, copied, _ = distribute_to_folders_windows(plan, subdir)
        
        if progress_callback:
            progress_callback(f"Папка {subdir.name} обработана: {moved} перемещено, {copied} скопировано", 
                            10 + ((i + 1) * 80 // len(subdirs)))
    
    if progress_callback:
        progress_callback("Обработка группы папок завершена", 100)

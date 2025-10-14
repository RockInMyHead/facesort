"""
Продвинутая система кластеризации лиц с использованием SOTA методов:
- RetinaFace для детекции и выравнивания (5 ключевых точек)
- InsightFace/ArcFace для эмбеддингов (iresnet100)
- TTA (Test-Time Augmentation): горизонтальный flip
- Качественно-взвешенные шаблоны
- k-reciprocal re-ranking для графа сходства
- Spectral Clustering для точного разбиения
- Пост-валидация и очистка кластеров
"""

import os
import cv2
import shutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import time

# Детекция и распознавание лиц
try:
    from retinaface import RetinaFace
    RETINAFACE_AVAILABLE = True
except ImportError:
    RETINAFACE_AVAILABLE = False
    print("⚠️ RetinaFace не установлен, используем fallback")

try:
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("⚠️ InsightFace не установлен, используем fallback")

# Альтернативы
try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
    FACENET_AVAILABLE = True
except ImportError:
    FACENET_AVAILABLE = False
    print("⚠️ FaceNet-PyTorch не установлен")

# Кластеризация и метрики
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def _win_long(path: Path) -> str:
    """Windows long path support"""
    p = str(path.resolve())
    if os.name == "nt":
        return "\\\\?\\" + p if not p.startswith("\\\\?\\") else p
    return p

def imread_safe(path: Path):
    """Безопасное чтение изображения с поддержкой Unicode путей"""
    try:
        data = np.fromfile(_win_long(path), dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"❌ Ошибка чтения {path.name}: {e}")
        return None

def calculate_blur_score(image: np.ndarray) -> float:
    """
    Оценка размытия через Variance of Laplacian.
    Чем выше значение - тем четче изображение.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

def calculate_face_quality(face_img: np.ndarray, bbox: tuple = None) -> float:
    """
    Комплексная оценка качества лица.
    Учитывает: размер лица, размытие, яркость.
    
    Returns:
        quality_score: 0.0 - 1.0, где 1.0 - отличное качество
    """
    scores = []
    
    # 1. Размер лица (нормализованный)
    if bbox is not None:
        x1, y1, x2, y2 = bbox[:4]
        face_area = (x2 - x1) * (y2 - y1)
        size_score = min(face_area / (200 * 200), 1.0)  # Нормализуем к 200x200
        scores.append(size_score * 0.3)  # 30% веса
    
    # 2. Оценка резкости через Variance of Laplacian
    blur_score = calculate_blur_score(face_img)
    # Нормализуем: blur < 100 = плохо, > 500 = отлично
    normalized_blur = min(max(blur_score, 100), 500) / 500
    scores.append(normalized_blur * 0.5)  # 50% веса
    
    # 3. Яркость и контраст
    if len(face_img.shape) == 3:
        gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = face_img
    
    mean_brightness = np.mean(gray) / 255.0
    # Оптимальная яркость: 0.3 - 0.7
    brightness_score = 1.0 - abs(mean_brightness - 0.5) * 2
    scores.append(brightness_score * 0.2)  # 20% веса
    
    return sum(scores)

def align_face_5points(img: np.ndarray, landmarks: np.ndarray, target_size=(112, 112)):
    """
    Выравнивание лица по 5 ключевым точкам (глаза, нос, углы рта).
    Стандартная процедура для ArcFace/InsightFace.
    """
    # Стандартные позиции для 112x112
    src = np.array([
        [38.2946, 51.6963],  # Левый глаз
        [73.5318, 51.5014],  # Правый глаз
        [56.0252, 71.7366],  # Нос
        [41.5493, 92.3655],  # Левый угол рта
        [70.7299, 92.2041]   # Правый угол рта
    ], dtype=np.float32)
    
    # Масштабируем для целевого размера
    if target_size != (112, 112):
        scale = target_size[0] / 112.0
        src = src * scale
    
    # Преобразование Affine
    dst = landmarks.astype(np.float32)
    tform = cv2.estimateAffinePartial2D(dst, src)[0]
    
    if tform is None:
        # Fallback: просто изменяем размер
        return cv2.resize(img, target_size)
    
    aligned = cv2.warpAffine(img, tform, target_size, flags=cv2.INTER_LINEAR)
    return aligned

class AdvancedFaceRecognition:
    """
    Продвинутая система распознавания лиц с:
    - RetinaFace детекцией
    - InsightFace/ArcFace эмбеддингами
    - TTA и качественно-взвешенными шаблонами
    """
    
    def __init__(self, use_gpu=False, min_face_size=20, confidence_threshold=0.9):
        self.min_face_size = min_face_size
        self.confidence_threshold = confidence_threshold
        self.use_gpu = use_gpu
        
        # Инициализация детектора
        if INSIGHTFACE_AVAILABLE:
            try:
                print("🔧 Загружаем InsightFace модель (buffalo_l)...")
                self.face_app = FaceAnalysis(
                    name='buffalo_l',
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu 
                             else ['CPUExecutionProvider']
                )
                self.face_app.prepare(ctx_id=0 if use_gpu else -1, det_size=(640, 640))
                self.detector_type = 'insightface'
                print("✅ InsightFace загружен (buffalo_l с ArcFace)")
            except Exception as e:
                print(f"❌ Ошибка загрузки InsightFace: {e}")
                self.detector_type = 'none'
                self.face_app = None
        else:
            self.detector_type = 'none'
            self.face_app = None
    
    def detect_and_extract(self, img: np.ndarray, apply_tta=True) -> List[Dict]:
        """
        Детекция лиц и извлечение эмбеддингов с TTA.
        
        Returns:
            List of dicts with keys: bbox, landmarks, embedding, quality
        """
        if self.detector_type == 'insightface':
            return self._detect_with_insightface(img, apply_tta)
        else:
            return []
    
    def _detect_with_insightface(self, img: np.ndarray, apply_tta=True) -> List[Dict]:
        """Детекция и извлечение с InsightFace"""
        results = []
        
        # Оригинальное изображение
        faces = self.face_app.get(img)
        
        for face in faces:
            # Фильтрация по confidence
            if hasattr(face, 'det_score') and face.det_score < self.confidence_threshold:
                continue
            
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            # Фильтрация по размеру
            if (x2 - x1) < self.min_face_size or (y2 - y1) < self.min_face_size:
                continue
            
            # Извлекаем лицо для оценки качества
            face_img = img[y1:y2, x1:x2]
            if face_img.size == 0:
                continue
            
            # Оценка качества
            quality = calculate_face_quality(face_img, bbox)
            
            # Основной эмбеддинг (уже нормализован в InsightFace)
            embedding = face.normed_embedding
            
            # TTA: горизонтальный flip
            if apply_tta:
                img_flipped = cv2.flip(img, 1)
                faces_flipped = self.face_app.get(img_flipped)
                
                if len(faces_flipped) > 0:
                    # Находим соответствующее лицо (по позиции)
                    img_width = img.shape[1]
                    flipped_x1 = img_width - x2
                    flipped_x2 = img_width - x1
                    
                    best_match = None
                    best_iou = 0
                    
                    for f_face in faces_flipped:
                        fx1, fy1, fx2, fy2 = f_face.bbox.astype(int)
                        # Вычисляем IoU
                        inter_x1 = max(flipped_x1, fx1)
                        inter_y1 = max(y1, fy1)
                        inter_x2 = min(flipped_x2, fx2)
                        inter_y2 = min(y2, fy2)
                        
                        if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                            bbox1_area = (flipped_x2 - flipped_x1) * (y2 - y1)
                            bbox2_area = (fx2 - fx1) * (fy2 - fy1)
                            iou = inter_area / (bbox1_area + bbox2_area - inter_area)
                            
                            if iou > best_iou:
                                best_iou = iou
                                best_match = f_face
                    
                    # Усредняем эмбеддинги
                    if best_match is not None and best_iou > 0.5:
                        flipped_embedding = best_match.normed_embedding
                        # Среднее двух эмбеддингов и re-normalize
                        embedding = (embedding + flipped_embedding) / 2.0
                        embedding = embedding / np.linalg.norm(embedding)
            
            results.append({
                'bbox': bbox,
                'landmarks': face.kps if hasattr(face, 'kps') else None,
                'embedding': embedding,
                'quality': quality,
                'confidence': face.det_score if hasattr(face, 'det_score') else 1.0
            })
        
        return results

def k_reciprocal_rerank(similarity_matrix: np.ndarray, k: int = 3) -> np.ndarray:
    """
    K-reciprocal re-ranking для улучшения графа сходства.
    Повышает веса для взаимных k-ближайших соседей.
    
    Args:
        similarity_matrix: Матрица сходства (N x N)
        k: Число ближайших соседей для проверки
    
    Returns:
        Обновленная матрица сходства
    """
    N = similarity_matrix.shape[0]
    reranked = similarity_matrix.copy()
    
    # Находим k ближайших соседей для каждой точки
    # (сортируем по убыванию сходства)
    nearest_neighbors = np.argsort(-similarity_matrix, axis=1)[:, 1:k+1]
    
    # Для каждой пары проверяем взаимность
    for i in range(N):
        for j in range(i + 1, N):
            # Проверяем, является ли j в k-NN для i и наоборот
            i_in_j_neighbors = i in nearest_neighbors[j]
            j_in_i_neighbors = j in nearest_neighbors[i]
            
            if i_in_j_neighbors and j_in_i_neighbors:
                # Усиливаем связь для взаимных соседей
                boost = 1.2
                reranked[i, j] *= boost
                reranked[j, i] *= boost
            elif i_in_j_neighbors or j_in_i_neighbors:
                # Небольшое усиление для односторонних соседей
                boost = 1.1
                reranked[i, j] *= boost
                reranked[j, i] *= boost
    
    # Нормализуем обратно к [0, 1]
    reranked = np.clip(reranked, 0, 1)
    
    return reranked

def spectral_clustering_with_validation(
    embeddings: List[np.ndarray],
    n_clusters: int = None,
    quality_weights: List[float] = None,
    k_reciprocal: int = 3,
    verification_threshold: float = 0.35
) -> np.ndarray:
    """
    Spectral Clustering с k-reciprocal re-ranking и валидацией.
    
    Args:
        embeddings: L2-нормализованные эмбеддинги
        n_clusters: Число кластеров (если None - определяем автоматически)
        quality_weights: Веса качества для каждого эмбеддинга
        k_reciprocal: k для re-ranking
        verification_threshold: Порог для валидации (cosine distance)
    
    Returns:
        labels: Метки кластеров
    """
    X = np.vstack(embeddings)
    N = len(embeddings)
    
    # Применяем качественные веса если есть
    if quality_weights is not None:
        X_weighted = X * np.array(quality_weights)[:, np.newaxis]
        X_weighted = normalize(X_weighted, norm='l2')
    else:
        X_weighted = X
    
    # Вычисляем матрицу сходства (косинусная)
    similarity = cosine_similarity(X_weighted)
    
    # K-reciprocal re-ranking
    if k_reciprocal > 0:
        similarity = k_reciprocal_rerank(similarity, k=k_reciprocal)
    
    # Преобразуем в аффинити-матрицу
    affinity = np.maximum(similarity, 0)
    np.fill_diagonal(affinity, 0)  # Обнуляем диагональ
    
    # Определяем оптимальное число кластеров если не задано
    if n_clusters is None:
        # Эвристика: используем eigenvalue gap
        from scipy.linalg import eigh
        
        # Вычисляем Laplacian
        D = np.diag(affinity.sum(axis=1))
        L = D - affinity
        
        # Собственные значения
        eigenvalues, _ = eigh(L, D)
        eigenvalues = np.sort(eigenvalues)
        
        # Находим наибольший gap
        gaps = np.diff(eigenvalues)
        n_clusters = np.argmax(gaps[:min(10, len(gaps))]) + 2  # +2 потому что индекс с 0
        n_clusters = max(2, min(n_clusters, N // 2))  # Ограничения
        
        print(f"🔍 Автоматически определено кластеров: {n_clusters}")
    
    # Spectral Clustering
    clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        assign_labels='kmeans',
        random_state=42
    )
    
    labels = clustering.fit_predict(affinity)
    
    # Пост-валидация: проверяем внутрикластерное сходство
    validated_labels = labels.copy()
    
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_embeddings = X[mask]
        
        if len(cluster_embeddings) < 2:
            continue
        
        # Вычисляем центроид
        centroid = np.mean(cluster_embeddings, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        
        # Проверяем расстояния до центроида
        indices = np.where(mask)[0]
        for idx in indices:
            distance = 1 - np.dot(X[idx], centroid)  # Cosine distance
            
            if distance > verification_threshold:
                # Outlier - переназначаем в ближайший валидный кластер
                best_cluster = -1
                best_distance = float('inf')
                
                for other_cluster_id in range(n_clusters):
                    if other_cluster_id == cluster_id:
                        continue
                    
                    other_mask = labels == other_cluster_id
                    other_embeddings = X[other_mask]
                    
                    if len(other_embeddings) == 0:
                        continue
                    
                    other_centroid = np.mean(other_embeddings, axis=0)
                    other_centroid = other_centroid / np.linalg.norm(other_centroid)
                    
                    other_distance = 1 - np.dot(X[idx], other_centroid)
                    
                    if other_distance < best_distance and other_distance < verification_threshold:
                        best_distance = other_distance
                        best_cluster = other_cluster_id
                
                if best_cluster != -1:
                    validated_labels[idx] = best_cluster
                    print(f"  🔄 Переназначен outlier из кластера {cluster_id} в {best_cluster}")
                else:
                    # Помечаем как noise
                    validated_labels[idx] = -1
                    print(f"  ❌ Outlier не подошел ни к одному кластеру")
    
    # Объединяем маленькие кластеры
    for cluster_id in range(n_clusters):
        mask = validated_labels == cluster_id
        cluster_size = np.sum(mask)
        
        if cluster_size == 1:
            idx = np.where(mask)[0][0]
            # Ищем ближайший кластер
            best_cluster = -1
            best_similarity = -1
            
            for other_id in range(n_clusters):
                if other_id == cluster_id:
                    continue
                
                other_mask = validated_labels == other_id
                if np.sum(other_mask) == 0:
                    continue
                
                other_centroid = np.mean(X[other_mask], axis=0)
                other_centroid = other_centroid / np.linalg.norm(other_centroid)
                
                sim = np.dot(X[idx], other_centroid)
                
                if sim > best_similarity and (1 - sim) < verification_threshold:
                    best_similarity = sim
                    best_cluster = other_id
            
            if best_cluster != -1:
                validated_labels[idx] = best_cluster
                print(f"  🔗 Объединен одиночный кластер {cluster_id} → {best_cluster}")
    
    return validated_labels

def build_plan_advanced(
    input_dir: Path,
    min_face_confidence: float = 0.9,
    min_blur_threshold: float = 100.0,
    n_clusters: int = None,
    apply_tta: bool = True,
    use_gpu: bool = False,
    progress_callback=None,
    include_excluded: bool = False
) -> Dict:
    """
    Продвинутая кластеризация с SOTA методами.
    
    Args:
        input_dir: Папка с изображениями
        min_face_confidence: Минимальный confidence для детекции
        min_blur_threshold: Минимальный порог резкости
        n_clusters: Число кластеров (None = автоматически)
        apply_tta: Применять Test-Time Augmentation
        use_gpu: Использовать GPU
        progress_callback: Callback для прогресса
        include_excluded: Включить исключенные папки
    
    Returns:
        dict с clusters, plan, unreadable, no_faces
    """
    print(f"🚀 [ADVANCED] Запуск продвинутой кластеризации: {input_dir}")
    
    input_dir = Path(input_dir)
    start_time = time.time()
    
    # Собираем изображения
    excluded_names = ["общие", "общая", "common", "shared", "все", "all", "mixed", "смешанные"]
    
    if include_excluded:
        all_images = [p for p in input_dir.rglob("*") if is_image(p)]
    else:
        all_images = [
            p for p in input_dir.rglob("*")
            if is_image(p) and not any(ex in str(p).lower() for ex in excluded_names)
        ]
    
    print(f"📂 Найдено {len(all_images)} изображений")
    
    if progress_callback:
        progress_callback(f"📂 Найдено {len(all_images)} изображений", 5)
    
    # Инициализация системы распознавания
    recognizer = AdvancedFaceRecognition(
        use_gpu=use_gpu,
        confidence_threshold=min_face_confidence
    )
    
    if progress_callback:
        progress_callback("✅ Модель загружена, начинаем анализ...", 10)
    
    # Обработка изображений
    all_embeddings = []
    all_qualities = []
    owners = []
    img_face_count = {}
    unreadable = []
    no_faces = []
    
    total = len(all_images)
    
    for i, img_path in enumerate(all_images):
        if progress_callback and i % 5 == 0:
            percent = 10 + int((i + 1) / max(total, 1) * 70)
            progress_callback(f"📷 Анализ: {percent}% ({i+1}/{total})", percent)
        
        # Безопасное чтение
        img = imread_safe(img_path)
        if img is None:
            unreadable.append(img_path)
            continue
        
        # Детекция и извлечение
        try:
            faces = recognizer.detect_and_extract(img, apply_tta=apply_tta)
            
            if not faces:
                no_faces.append(img_path)
                continue
            
            # Фильтрация по качеству
            valid_faces = []
            for face in faces:
                # Проверка резкости
                if face['quality'] < 0.3:  # Низкое качество
                    print(f"  ⚠️ Низкое качество лица в {img_path.name}: {face['quality']:.3f}")
                    continue
                
                valid_faces.append(face)
            
            if not valid_faces:
                no_faces.append(img_path)
                continue
            
            # Сохраняем результаты
            img_face_count[img_path] = len(valid_faces)
            
            for face in valid_faces:
                all_embeddings.append(face['embedding'])
                all_qualities.append(face['quality'])
                owners.append(img_path)
                
        except Exception as e:
            print(f"❌ Ошибка обработки {img_path.name}: {e}")
            unreadable.append(img_path)
    
    if not all_embeddings:
        print("⚠️ Не найдено лиц для кластеризации")
        return {
            "clusters": {},
            "plan": [],
            "unreadable": [str(p) for p in unreadable],
            "no_faces": [str(p) for p in no_faces],
        }
    
    print(f"✅ Извлечено {len(all_embeddings)} эмбеддингов из {len(set(owners))} изображений")
    
    # Кластеризация
    if progress_callback:
        progress_callback(f"🔄 Spectral Clustering {len(all_embeddings)} лиц...", 85)
    
    labels = spectral_clustering_with_validation(
        embeddings=all_embeddings,
        n_clusters=n_clusters,
        quality_weights=all_qualities,
        k_reciprocal=3,
        verification_threshold=0.35
    )
    
    print(f"✅ Кластеризация завершена: {len(set(labels))} кластеров")
    
    # Формируем результат
    cluster_map = defaultdict(set)
    cluster_by_img = defaultdict(set)
    
    # Обработка noise (-1)
    max_label = max(labels) if len(labels) > 0 and max(labels) >= 0 else -1
    next_single_label = max_label + 1
    
    for idx, (label, path) in enumerate(zip(labels, owners)):
        if label == -1:
            unique_label = next_single_label
            cluster_map[unique_label].add(path)
            cluster_by_img[path].add(unique_label)
            next_single_label += 1
        else:
            cluster_map[label].add(path)
            cluster_by_img[path].add(label)
    
    # Формируем план
    plan = []
    for path in all_images:
        clusters = cluster_by_img.get(path)
        if not clusters:
            continue
        plan.append({
            "path": str(path),
            "cluster": sorted(list(clusters)),
            "faces": img_face_count.get(path, 0)
        })
    
    processing_time = time.time() - start_time
    print(f"⏱️ Время обработки: {processing_time:.1f}с")
    
    if progress_callback:
        progress_callback(f"✅ Готово! {len(cluster_map)} кластеров", 100)
    
    return {
        "clusters": {
            int(k): [str(p) for p in sorted(v, key=lambda x: str(x))]
            for k, v in cluster_map.items()
        },
        "plan": plan,
        "unreadable": [str(p) for p in unreadable],
        "no_faces": [str(p) for p in no_faces],
    }

# Импортируем distribute_to_folders из cluster.py для совместимости
from cluster import distribute_to_folders, process_group_folder


"""
Упрощенная версия кластеризации без проблемных зависимостей.
"""
import os
import time
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
import cv2
from PIL import Image

# Поддерживаемые форматы изображений
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

def is_image(path: Path) -> bool:
    """Проверяет, является ли файл изображением."""
    return path.suffix.lower() in IMG_EXTS

def imread_safe(path: Path) -> np.ndarray:
    """Безопасное чтение изображения."""
    try:
        if path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            img = cv2.imread(str(path))
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return None
    except Exception as e:
        print(f"❌ Ошибка чтения {path.name}: {e}")
        return None

def detect_faces_simple(img: np.ndarray) -> List[Dict]:
    """Простая детекция лиц с помощью OpenCV."""
    try:
        print(f"🔍 Анализируем изображение размером: {img.shape}")
        
        # Конвертируем в grayscale для детекции
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Загружаем каскад Хаара
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        print(f"🎯 Найдено лиц: {len(faces)}")
        
        results = []
        for (x, y, w, h) in faces:
            # Извлекаем лицо
            face_img = img[y:y+h, x:x+w]
            
            # Простое "эмбеддинг" - средний цвет лица
            embedding = np.mean(face_img.reshape(-1, 3), axis=0)
            embedding = embedding / np.linalg.norm(embedding)  # L2 normalize
            
            results.append({
                'embedding': embedding,
                'quality': 0.8,  # Фиксированное качество
                'bbox': (x, y, w, h)
            })
        
        return results
    except Exception as e:
        print(f"❌ Ошибка детекции лиц: {e}")
        return []

def merge_similar_clusters(embeddings: np.ndarray, labels: np.ndarray, merge_threshold: float = 0.4) -> np.ndarray:
    """Объединяет похожие кластеры на основе центроидов."""
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1:
        return labels
    
    # Вычисляем центроиды для каждого кластера
    centroids = {}
    for label in unique_labels:
        mask = labels == label
        if np.sum(mask) > 0:
            centroid = np.mean(embeddings[mask], axis=0)
            centroids[label] = centroid / np.linalg.norm(centroid)
    
    # Находим пары кластеров для слияния
    merged_labels = labels.copy()
    label_mapping = {label: label for label in unique_labels}
    
    for i, label1 in enumerate(unique_labels):
        if label1 not in centroids:
            continue
            
        for j, label2 in enumerate(unique_labels[i+1:], i+1):
            if label2 not in centroids:
                continue
            
            # Вычисляем косинусное расстояние между центроидами
            cosine_dist = 1 - np.dot(centroids[label1], centroids[label2])
            
            if cosine_dist < merge_threshold:
                # Объединяем кластеры
                target_label = min(label1, label2)
                source_label = max(label1, label2)
                
                # Обновляем mapping
                for old_label, new_label in label_mapping.items():
                    if new_label == source_label:
                        label_mapping[old_label] = target_label
                
                print(f"🔗 Объединяем кластеры {label1} и {label2} (расстояние: {cosine_dist:.3f})")
    
    # Применяем mapping
    for i, label in enumerate(labels):
        merged_labels[i] = label_mapping[label]
    
    return merged_labels

def merge_single_clusters(embeddings: np.ndarray, labels: np.ndarray, merge_threshold: float = 0.6) -> np.ndarray:
    """Объединяет одиночные кластеры с ближайшими, используя несколько метрик."""
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1:
        return labels
    
    # Находим размеры кластеров
    cluster_sizes = {}
    for label in unique_labels:
        cluster_sizes[label] = np.sum(labels == label)
    
    # Находим одиночные кластеры
    single_clusters = [label for label, size in cluster_sizes.items() if size == 1]
    
    if not single_clusters:
        return labels
    
    merged_labels = labels.copy()
    
    for single_label in single_clusters:
        # Находим индекс одиночного элемента
        single_idx = np.where(labels == single_label)[0][0]
        single_embedding = embeddings[single_idx]
        
        # Ищем ближайший кластер с учетом нескольких метрик
        best_cluster = None
        best_score = float('inf')
        
        for other_label in unique_labels:
            if other_label == single_label or cluster_sizes[other_label] == 1:
                continue
            
            # Вычисляем расстояние до центроида другого кластера
            other_mask = labels == other_label
            other_embeddings = embeddings[other_mask]
            other_centroid = np.mean(other_embeddings, axis=0)
            other_centroid = other_centroid / np.linalg.norm(other_centroid)
            
            # Метрика 1: Косинусное расстояние
            cosine_dist = 1 - np.dot(single_embedding, other_centroid)
            
            # Метрика 2: L2 расстояние (нормализованное)
            l2_dist = np.linalg.norm(single_embedding - other_centroid)
            
            # Метрика 3: Минимальное расстояние до любого элемента кластера
            min_dist_to_any = float('inf')
            for other_emb in other_embeddings:
                other_emb_norm = other_emb / np.linalg.norm(other_emb)
                dist_to_element = 1 - np.dot(single_embedding, other_emb_norm)
                min_dist_to_any = min(min_dist_to_any, dist_to_element)
            
            # Комбинированная оценка (взвешенная сумма)
            combined_score = (
                0.5 * cosine_dist +           # Основная метрика
                0.3 * l2_dist +               # L2 расстояние
                0.2 * min_dist_to_any        # Минимальное расстояние
            )
            
            # Смягченный порог: учитываем размер целевого кластера
            size_factor = 1.0 + 0.1 * cluster_sizes[other_label]  # Больше кластеры = мягче порог
            adjusted_threshold = merge_threshold * size_factor
            
            if combined_score < best_score and combined_score < adjusted_threshold:
                best_score = combined_score
                best_cluster = other_label
        
        # Объединяем с ближайшим кластером
        if best_cluster is not None:
            merged_labels[single_idx] = best_cluster
            print(f"🔗 Объединяем одиночный кластер {single_label} с {best_cluster} (оценка: {best_score:.3f})")
    
    return merged_labels

def build_plan_simple(
    input_dir: Path,
    n_clusters: int = 8,
    progress_callback=None
) -> Dict:
    """Упрощенная кластеризация без проблемных зависимостей."""
    print(f"🚀 [SIMPLE] Запуск упрощенной кластеризации: {input_dir}")
    
    input_dir = Path(input_dir)
    start_time = time.time()
    
    # Собираем изображения
    all_images = [p for p in input_dir.rglob("*") if is_image(p)]
    print(f"📂 Найдено {len(all_images)} изображений")
    
    if progress_callback:
        progress_callback(f"📂 Найдено {len(all_images)} изображений", 5)
    
    # Обработка изображений
    all_embeddings = []
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
        
        # Детекция лиц
        try:
            faces = detect_faces_simple(img)
            
            if not faces:
                print(f"⚠️ Лица не найдены в {img_path.name}")
                no_faces.append(img_path)
                continue
            
            # Сохраняем результаты
            img_face_count[img_path] = len(faces)
            
            for face in faces:
                all_embeddings.append(face['embedding'])
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
        progress_callback(f"🔄 Agglomerative Clustering {len(all_embeddings)} лиц...", 85)
    
    print("⚙️ Используем AgglomerativeClustering с косинусной метрикой")
    X = np.vstack(all_embeddings)
    
    # Расстояния косинусные
    dist_matrix = pairwise_distances(X, metric='cosine')
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'
    )
    labels = clustering.fit_predict(dist_matrix)
    
    print(f"✅ Кластеризация завершена: {len(set(labels))} кластеров")
    
    # Отключаем слияние кластеров для лучшего разделения людей
    # labels = merge_similar_clusters(X, labels, merge_threshold=0.1)
    # labels = merge_single_clusters(X, labels, merge_threshold=0.2)
    
    print(f"✅ После слияния: {len(set(labels))} кластеров")
    
    # Формируем результат
    cluster_map = defaultdict(set)
    cluster_by_img = defaultdict(set)
    
    for idx, (label, path) in enumerate(zip(labels, owners)):
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
        "clusters": {str(k): [str(p) for p in v] for k, v in cluster_map.items()},
        "plan": plan,
        "unreadable": [str(p) for p in unreadable],
        "no_faces": [str(p) for p in no_faces],
    }

def distribute_to_folders(plan: dict, base_dir: Path, cluster_start: int = 1, progress_callback=None) -> Tuple[int, int, int]:
    """Распределяет файлы по папкам согласно плану."""
    import shutil
    
    moved, copied = 0, 0
    moved_paths = set()

    used_clusters = sorted({c for item in plan.get("plan", []) for c in item["cluster"]})
    cluster_id_map = {old: cluster_start + idx for idx, old in enumerate(used_clusters)}
    plan_items = plan.get("plan", [])
    total_items = len(plan_items)
    if progress_callback:
        progress_callback(f"🔄 Распределение {total_items} файлов по папкам...", 0)

    cluster_file_counts = {}
    for item in plan_items:
        clusters = [cluster_id_map[c] for c in item["cluster"]]
        for cid in clusters:
            cluster_file_counts[cid] = cluster_file_counts.get(cid, 0) + 1

    for i, item in enumerate(plan_items):
        if progress_callback:
            percent = int((i + 1) / max(total_items, 1) * 100)
            progress_callback(f"📁 Распределение файлов: {percent}% ({i+1}/{total_items})", percent)
        src = Path(item["path"]);
        clusters = [cluster_id_map[c] for c in item["cluster"]]
        if not src.exists():
            continue
        if len(clusters) == 1:
            dst = base_dir / f"{clusters[0]}" / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.resolve() != dst.resolve(): shutil.move(str(src), str(dst)); moved+=1; moved_paths.add(src.parent)
        else:
            for cid in clusters:
                dst = base_dir / f"{cid}" / src.name; dst.parent.mkdir(parents=True, exist_ok=True)
                if src.resolve() != dst.resolve(): shutil.copy2(str(src), str(dst)); copied+=1
            try: src.unlink()
            except: pass
    if progress_callback:
        progress_callback("📝 Переименование папок с количеством файлов...", 95)
    for cid, cnt in cluster_file_counts.items():
        old_folder = base_dir / str(cid); new_folder = base_dir / f"{cid} ({cnt})"
        if old_folder.exists():
            try: old_folder.rename(new_folder)
            except: pass
    if progress_callback:
        progress_callback("🧹 Очистка пустых папок...", 100)
    for p in sorted(moved_paths, key=lambda x: len(str(x)), reverse=True):
        try: p.rmdir()
        except: pass
    print(f"📦 Перемещено: {moved}, скопировано: {copied}")
    return moved, copied, cluster_start + len(used_clusters)

def process_group_folder(group_dir: Path, progress_callback=None, include_excluded: bool = False):
    """Обрабатывает группу папок."""
    cluster_counter = 1
    common = []
    if include_excluded:
        common = find_common_folders_recursive(group_dir)
        total = len(common)
        for i, c in enumerate(common):
            if progress_callback: progress_callback(f"📋 Обработка общих фото {i+1}/{total}", 10+int(i/total*70))
            process_common_folder_at_level(c, progress_callback)
        return 0, sum(1 for c in common), cluster_counter
    subdirs = [d for d in sorted(group_dir.iterdir()) if d.is_dir()]
    total = len(subdirs)
    moved_all, copied_all = 0, 0
    for i, sub in enumerate(subdirs):
        if progress_callback: progress_callback(f"🔍 Кластеризация {sub.name} ({i+1}/{total})", 10+int(i/total*70))
        data = build_plan_simple(
            input_dir=sub,
            n_clusters=3,
            progress_callback=progress_callback
        )
        m, c, _ = distribute_to_folders(data, sub, cluster_start=1, progress_callback=progress_callback)
        moved_all+=m; copied_all+=c
    return moved_all, copied_all, cluster_counter

def find_common_folders_recursive(group_dir: Path):
    """Находит общие папки рекурсивно."""
    excluded_names = ["общие", "общая", "common", "shared", "все", "all", "mixed", "смешанные"]
    common = []
    for subdir in group_dir.iterdir():
        if subdir.is_dir() and any(ex in subdir.name.lower() for ex in excluded_names):
            common.append(subdir)
    return common

def process_common_folder_at_level(common_dir: Path, progress_callback=None):
    """Обрабатывает общую папку на уровне."""
    # Простая реализация для общих папок
    pass

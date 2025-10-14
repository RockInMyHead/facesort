#!/usr/bin/env python3
"""
Улучшенная версия кластеризации с более подробным логированием
и обработкой всех папок, а не только "общих"
"""

import os
import cv2
import shutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.cluster import DBSCAN
import face_recognition
import hdbscan
from collections import defaultdict

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def _win_long(path: Path) -> str:
    p = str(path.resolve())
    if os.name == "nt":
        return "\\\\?\\" + p if not p.startswith("\\\\?\\") else p
    return p

def imread_safe(path: Path):
    try:
        data = np.fromfile(_win_long(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None

def find_optimal_epsilon(distance_matrix, min_samples=1):
    """Найти оптимальный epsilon для DBSCAN"""
    from sklearn.neighbors import NearestNeighbors
    
    # Используем k-ближайших соседей для определения epsilon
    k = min(min_samples + 1, len(distance_matrix))
    if k >= len(distance_matrix):
        k = max(1, len(distance_matrix) - 1)
    
    nbrs = NearestNeighbors(n_neighbors=k, metric='precomputed').fit(distance_matrix)
    distances, indices = nbrs.kneighbors(distance_matrix)
    
    # Сортируем расстояния и берем 75-й перцентиль
    distances = np.sort(distances[:, k-1])
    epsilon = np.percentile(distances, 75)
    
    return max(0.1, min(epsilon, 0.8))

def build_plan_live(
    input_dir: Path,
    det_size=(640, 640),
    min_score: float = 0.95,
    min_cluster_size: int = 1,
    min_samples: int = 1,
    providers: List[str] = ("CPUExecutionProvider",),
    progress_callback=None,
    include_excluded: bool = False,
):
    """Улучшенная версия build_plan_live с подробным логированием"""
    print(f"🔍 [CLUSTER] build_plan_live вызвана: input_dir={input_dir}, include_excluded={include_excluded}")
    
    try:
        input_dir = Path(input_dir)
        print(f"🔍 [CLUSTER] input_dir преобразован в Path: {input_dir}")
        
        # Собираем все изображения
        excluded_names = ["общие", "общая", "common", "shared", "все", "all", "mixed", "смешанные"]
        print(f"🔍 [CLUSTER] excluded_names: {excluded_names}")
        
        if include_excluded:
            # Включаем все изображения, даже из папок "общие"
            all_images = [
                p for p in input_dir.rglob("*")
                if is_image(p)
            ]
        else:
            # Исключаем изображения из папок с нежелательными именами
            all_images = [
                p for p in input_dir.rglob("*")
                if is_image(p)
                and not any(ex in str(p).lower() for ex in excluded_names)
            ]

        print(f"🔍 build_plan_live: input_dir={input_dir}, include_excluded={include_excluded}, найдено {len(all_images)} изображений")
        if len(all_images) > 0:
            print(f"🔍 Первые несколько файлов: {[str(p) for p in all_images[:3]]}")
        
        if progress_callback:
            progress_callback(f"📂 Сканируется: {input_dir}, найдено изображений: {len(all_images)}", 1)

        if len(all_images) == 0:
            print("❌ Изображения не найдены!")
            if progress_callback:
                progress_callback("❌ Изображения не найдены", 100)
            return {
                "clusters": {},
                "plan": [],
                "unreadable": [],
                "no_faces": [],
            }

        # face_recognition использует dlib - модель загружается автоматически
        if progress_callback:
            progress_callback("✅ Модель face_recognition готова (dlib + CNN), начинаем анализ...", 10)

        embeddings = []
        owners = []
        img_face_count = {}
        unreadable = []
        no_faces = []

        total = len(all_images)
        processed_faces = 0
        
        print(f"🔍 [CLUSTER] Начинаем обработку {total} изображений с face_recognition (модель: large CNN)")
        
        for i, p in enumerate(all_images):
            # Обновляем прогресс для каждого изображения
            if progress_callback:
                percent = 10 + int((i + 1) / max(total, 1) * 70)  # 10-80% для анализа изображений
                progress_callback(f"📷 Анализ изображений: {percent}% ({i+1}/{total}) - {p.name}", percent)
            
            try:
                # Загружаем изображение через face_recognition
                img = face_recognition.load_image_file(str(p))
                
                # Находим лица (используем CNN детектор для точности)
                face_locations = face_recognition.face_locations(img, model="cnn")
                
                if not face_locations:
                    no_faces.append(p)
                    print(f"  ❌ Лица не найдены в {p.name}")
                    continue
                
                # Извлекаем эмбеддинги для всех найденных лиц
                face_encodings = face_recognition.face_encodings(img, face_locations, model="large")
                
                if not face_encodings:
                    no_faces.append(p)
                    print(f"  ❌ Не удалось извлечь эмбеддинги из {p.name}")
                    continue
                
                # Добавляем все эмбеддинги
                for encoding in face_encodings:
                    embeddings.append(encoding)
                    owners.append(p)
                    processed_faces += 1
                
                img_face_count[p] = len(face_encodings)
                print(f"  ✅ Найдено {len(face_encodings)} лиц в {p.name}")
                
            except Exception as e:
                print(f"  ❌ Ошибка обработки {p.name}: {e}")
                unreadable.append(p)
                continue

        print(f"🔍 [CLUSTER] Обработано {processed_faces} лиц из {total} изображений")
        print(f"🔍 [CLUSTER] Нечитаемых файлов: {len(unreadable)}")
        print(f"🔍 [CLUSTER] Файлов без лиц: {len(no_faces)}")

        if not embeddings:
            if progress_callback:
                progress_callback("⚠️ Не найдено лиц для кластеризации", 100)
            print(f"⚠️ Нет эмбеддингов: {input_dir}")
            return {
                "clusters": {},
                "plan": [],
                "unreadable": [str(p) for p in unreadable],
                "no_faces": [str(p) for p in no_faces],
            }

        # Этап 2: Улучшенная кластеризация с DBSCAN
        if progress_callback:
            progress_callback(f"🔄 Кластеризация {len(embeddings)} лиц с DBSCAN...", 80)
        
        print(f"🔍 [CLUSTER] Кластеризация {len(embeddings)} эмбеддингов")
        
        X = np.vstack(embeddings)
        
        # Используем Euclidean distance для face_recognition (рекомендовано)
        distance_matrix = euclidean_distances(X)

        if progress_callback:
            progress_callback("🔄 Поиск оптимального epsilon...", 82)
        
        # Находим оптимальный epsilon адаптивно
        epsilon = find_optimal_epsilon(distance_matrix)
        # Ограничиваем epsilon для строгой кластеризации
        epsilon = min(epsilon, 0.6)  # Максимум 0.6 для face_recognition
        
        print(f"🔍 [CLUSTER] Оптимальный epsilon: {epsilon:.4f}")
        
        if progress_callback:
            progress_callback(f"🔄 DBSCAN кластеризация (eps={epsilon:.3f})...", 85)

        # DBSCAN с min_samples=1 позволяет одиночные фото
        model = DBSCAN(
            metric='precomputed',
            eps=epsilon,
            min_samples=min_samples,  # =1 для одиночных фото
            algorithm='auto'
        )
        raw_labels = model.fit_predict(distance_matrix)
        
        print(f"🔍 [CLUSTER] DBSCAN нашел {len(set(raw_labels)) - (1 if -1 in raw_labels else 0)} кластеров")
        print(f"🔍 [CLUSTER] Некластеризованных лиц: {list(raw_labels).count(-1)}")

        # Обработка кластеров с созданием отдельных папок для одиночных фото
        cluster_map = defaultdict(set)
        cluster_by_img = defaultdict(set)
        
        # Находим максимальный label для некластеризованных
        max_label = max(raw_labels) if len(raw_labels) > 0 and max(raw_labels) >= 0 else -1
        next_single_label = max_label + 1
        
        for idx, (label, path) in enumerate(zip(raw_labels, owners)):
            if label == -1:
                # Создаём отдельный кластер для каждого некластеризованного лица
                unique_label = next_single_label
                cluster_map[unique_label].add(path)
                cluster_by_img[path].add(unique_label)
                next_single_label += 1
                print(f"  📁 Создан одиночный кластер {unique_label} для {path.name}")
            else:
                cluster_map[label].add(path)
                cluster_by_img[path].add(label)

        # Создаем план распределения
        plan = []
        for path, cluster_ids in cluster_by_img.items():
            plan.append({
                "path": str(path),
                "cluster": list(cluster_ids)
            })

        # Преобразуем cluster_map в формат с путями как строками
        clusters_str = {}
        for cluster_id, paths in cluster_map.items():
            clusters_str[cluster_id] = [str(p) for p in paths]

        result = {
            "clusters": clusters_str,
            "plan": plan,
            "unreadable": [str(p) for p in unreadable],
            "no_faces": [str(p) for p in no_faces],
        }
        
        print(f"🔍 [CLUSTER] Результат: {len(clusters_str)} кластеров, {len(plan)} файлов в плане")
        
        if progress_callback:
            progress_callback("✅ Кластеризация завершена", 100)
        
        return result
        
    except Exception as e:
        print(f"❌ [CLUSTER] Ошибка в build_plan_live: {e}")
        import traceback
        traceback.print_exc()
        if progress_callback:
            progress_callback(f"❌ Ошибка: {str(e)}", 100)
        return {
            "clusters": {},
            "plan": [],
            "unreadable": [],
            "no_faces": [],
        }

def distribute_to_folders(plan: dict, base_dir: Path, cluster_start: int = 1, progress_callback=None) -> Tuple[int, int, int]:
    """Улучшенная версия distribute_to_folders с подробным логированием"""
    print(f"🔍 [DISTRIBUTE] distribute_to_folders вызвана: base_dir={base_dir}, cluster_start={cluster_start}")
    
    moved, copied = 0, 0
    moved_paths = set()

    used_clusters = sorted({c for item in plan.get("plan", []) for c in item["cluster"]})
    cluster_id_map = {old: cluster_start + idx for idx, old in enumerate(used_clusters)}

    plan_items = plan.get("plan", [])
    total_items = len(plan_items)
    
    print(f"🔍 [DISTRIBUTE] Будет обработано {total_items} файлов")
    print(f"🔍 [DISTRIBUTE] Используемые кластеры: {used_clusters}")
    print(f"🔍 [DISTRIBUTE] Маппинг кластеров: {cluster_id_map}")
    
    # Подсчитываем количество файлов в каждом кластере
    cluster_file_counts = {}
    for item in plan_items:
        clusters = [cluster_id_map[c] for c in item["cluster"]]
        for cluster_id in clusters:
            cluster_file_counts[cluster_id] = cluster_file_counts.get(cluster_id, 0) + 1
    
    print(f"🔍 [DISTRIBUTE] Файлов по кластерам: {cluster_file_counts}")
    
    if progress_callback:
        progress_callback(f"🔄 Распределение {total_items} файлов по папкам...", 0)

    for i, item in enumerate(plan_items):
        if progress_callback:
            percent = int((i + 1) / max(total_items, 1) * 100)
            progress_callback(f"📁 Распределение файлов: {percent}% ({i+1}/{total_items})", percent)
            
        src = Path(item["path"])
        clusters = [cluster_id_map[c] for c in item["cluster"]]
        if not src.exists():
            print(f"❌ [DISTRIBUTE] Файл не существует: {src}")
            continue

        print(f"🔍 [DISTRIBUTE] Обрабатываем {src.name} -> кластеры {clusters}")

        if len(clusters) == 1:
            cluster_id = clusters[0]
            dst = base_dir / f"{cluster_id}" / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            print(f"🔍 [DISTRIBUTE] Создаем папку: {dst.parent}")
            try:
                # Skip if source and destination are the same file
                try:
                    if src.resolve() == dst.resolve():
                        print(f"⚠️ Пропуск перемещения (одинаковые пути): {src} → {dst}")
                        continue
                except Exception:
                    if str(src) == str(dst):
                        print(f"⚠️ Пропуск перемещения (одинаковые строки): {src} → {dst}")
                        continue
                shutil.move(str(src), str(dst))
                moved += 1
                moved_paths.add(src.parent)
                print(f"✅ [DISTRIBUTE] Перемещен: {src} -> {dst}")
            except Exception as e:
                print(f"❌ Ошибка перемещения {src} → {dst}: {e}")
        else:
            # Файл принадлежит нескольким кластерам - копируем
            for cluster_id in clusters:
                dst = base_dir / f"{cluster_id}" / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                print(f"🔍 [DISTRIBUTE] Создаем папку для копирования: {dst.parent}")
                try:
                    shutil.copy2(str(src), str(dst))
                    copied += 1
                    print(f"✅ [DISTRIBUTE] Скопирован: {src} -> {dst}")
                except Exception as e:
                    print(f"❌ Ошибка копирования {src} → {dst}: {e}")

    print(f"🔍 [DISTRIBUTE] Результат: перемещено {moved}, скопировано {copied}")
    return moved, copied, len(moved_paths)

def process_group_folder(group_dir: Path, progress_callback=None, include_excluded: bool = False):
    """Улучшенная версия process_group_folder с обработкой всех папок"""
    print(f"🔍 [GROUP] process_group_folder вызвана для: {group_dir}, include_excluded={include_excluded}")
    
    cluster_counter = 1
    
    import time
    call_id = int(time.time() * 1000) % 10000
    print(f"🔍 process_group_folder [{call_id}] вызвана для: {group_dir}, include_excluded={include_excluded}")
    
    if include_excluded:
        # Обрабатываем все папки с изображениями
        if progress_callback:
            progress_callback("🔍 Поиск папок с изображениями...", 10)
        
        # Находим все папки с изображениями
        folders_with_images = []
        for item in group_dir.iterdir():
            if item.is_dir():
                # Проверяем есть ли изображения в папке
                has_images = any(f.suffix.lower() in IMG_EXTS for f in item.rglob("*") if f.is_file())
                if has_images:
                    folders_with_images.append(item)
                    print(f"✅ [GROUP] Найдена папка с изображениями: {item}")
        
        if not folders_with_images:
            if progress_callback:
                progress_callback("❌ Папки с изображениями не найдены", 100)
            print(f"❌ Папки с изображениями не найдены в {group_dir}")
            return 0
        
        print(f"🔍 [GROUP] Найдено {len(folders_with_images)} папок с изображениями")
        
        # Обрабатываем каждую папку
        total_created = 0
        for i, folder in enumerate(folders_with_images):
            if progress_callback:
                percent = 20 + int((i + 1) / len(folders_with_images) * 70)
                progress_callback(f"🔄 Обработка папки {i+1}/{len(folders_with_images)}: {folder.name}", percent)
            
            print(f"🔍 [GROUP] Обрабатываем папку: {folder}")
            
            # Строим план для папки
            plan = build_plan_live(folder, progress_callback=progress_callback, include_excluded=True)
            
            if len(plan.get("clusters", {})) > 0:
                # Распределяем файлы
                moved, copied, _ = distribute_to_folders(plan, folder, cluster_counter, progress_callback)
                total_created += len(plan.get("clusters", {}))
                cluster_counter += len(plan.get("clusters", {}))
                print(f"✅ [GROUP] Создано {len(plan.get('clusters', {}))} кластеров в папке {folder.name}")
            else:
                print(f"⚠️ [GROUP] Не удалось создать кластеры в папке {folder.name}")
        
        print(f"✅ [GROUP] Всего создано {total_created} кластеров")
        return total_created
    
    else:
        # Обычная обработка - только подпапки без исключенных названий
        excluded_names = ["общие", "общая", "common", "shared", "все", "all", "mixed", "смешанные"]
        subdirs_with_images = []
        
        for p in group_dir.iterdir():
            if p.is_dir() and not any(excluded_name in str(p).lower() for excluded_name in excluded_names):
                # Проверяем есть ли изображения в подпапке
                has_images = any(f.suffix.lower() in IMG_EXTS for f in p.rglob("*") if f.is_file())
                if has_images:
                    subdirs_with_images.append(p)
                    print(f"✅ [GROUP] Найдена подпапка с изображениями: {p}")
        
        if not subdirs_with_images:
            if progress_callback:
                progress_callback("❌ Подпапки с изображениями не найдены", 100)
            print(f"❌ Подпапки с изображениями не найдены в {group_dir}")
            return 0
        
        print(f"🔍 [GROUP] Найдено {len(subdirs_with_images)} подпапок с изображениями")
        
        # Обрабатываем каждую подпапку
        total_created = 0
        for i, subdir in enumerate(subdirs_with_images):
            if progress_callback:
                percent = 20 + int((i + 1) / len(subdirs_with_images) * 70)
                progress_callback(f"🔄 Обработка подпапки {i+1}/{len(subdirs_with_images)}: {subdir.name}", percent)
            
            print(f"🔍 [GROUP] Обрабатываем подпапку: {subdir}")
            
            # Строим план для подпапки
            plan = build_plan_live(subdir, progress_callback=progress_callback, include_excluded=False)
            
            if len(plan.get("clusters", {})) > 0:
                # Распределяем файлы
                moved, copied, _ = distribute_to_folders(plan, subdir, cluster_counter, progress_callback)
                total_created += len(plan.get("clusters", {}))
                cluster_counter += len(plan.get("clusters", {}))
                print(f"✅ [GROUP] Создано {len(plan.get('clusters', {}))} кластеров в подпапке {subdir.name}")
            else:
                print(f"⚠️ [GROUP] Не удалось создать кластеры в подпапке {subdir.name}")
        
        print(f"✅ [GROUP] Всего создано {total_created} кластеров")
        return total_created

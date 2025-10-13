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

def merge_clusters_by_centroid(
    embeddings: List[np.ndarray],
    owners: List[Path],
    raw_labels: np.ndarray,
    threshold: Optional[float] = None,
    auto_threshold: bool = False,
    margin: float = 0.05,
    min_threshold: float = 0.2,
    max_threshold: float = 0.4,
    progress_callback=None
) -> Tuple[Dict[int, Set[Path]], Dict[Path, Set[int]]]:

    if progress_callback:
        progress_callback("🔄 Объединение близких кластеров...", 92)

    cluster_embeddings: Dict[int, List[np.ndarray]] = defaultdict(list)
    cluster_paths: Dict[int, List[Path]] = defaultdict(list)

    for label, emb, path in zip(raw_labels, embeddings, owners):
        if label == -1:
            continue
        cluster_embeddings[label].append(emb)
        cluster_paths[label].append(path)

    centroids = {label: np.mean(embs, axis=0) for label, embs in cluster_embeddings.items()}
    labels = list(centroids.keys())

    if auto_threshold and threshold is None:
        pairwise = [cosine_distances([centroids[a]], [centroids[b]])[0][0]
                    for i, a in enumerate(labels) for b in labels[i+1:]]
        if pairwise:
            mean_dist = np.mean(pairwise)
            threshold = max(min_threshold, min(mean_dist - margin, max_threshold))
        else:
            threshold = min_threshold

        if progress_callback:
            progress_callback(f"📏 Авто-порог объединения: {threshold:.3f}", 93)
    elif threshold is None:
        threshold = 0.3

    next_cluster_id = 0
    label_to_group = {}
    total = len(labels)

    for i, label_i in enumerate(labels):
        if progress_callback:
            percent = 93 + int((i + 1) / max(total, 1) * 2)
            progress_callback(f"🔁 Слияние кластеров: {percent}% ({i+1}/{total})", percent)

        if label_i in label_to_group:
            continue
        group = [label_i]
        for j in range(i + 1, len(labels)):
            label_j = labels[j]
            if label_j in label_to_group:
                continue
            dist = cosine_distances([centroids[label_i]], [centroids[label_j]])[0][0]
            if dist < threshold:
                group.append(label_j)

        for l in group:
            label_to_group[l] = next_cluster_id
        next_cluster_id += 1

    merged_clusters: Dict[int, Set[Path]] = defaultdict(set)
    cluster_by_img: Dict[Path, Set[int]] = defaultdict(set)

    for label, path in zip(raw_labels, owners):
        if label == -1:
            continue
        new_label = label_to_group[label]
        merged_clusters[new_label].add(path)
        cluster_by_img[path].add(new_label)

    return merged_clusters, cluster_by_img

def find_optimal_epsilon(distance_matrix):
    """
    Находит оптимальный epsilon для DBSCAN через анализ k-distance графика.
    Использует 75-й перцентиль расстояний до ближайшего соседа.
    """
    distances = []
    for i in range(len(distance_matrix)):
        row = distance_matrix[i]
        sorted_dist = np.sort(row)
        if len(sorted_dist) > 1:
            # Расстояние до ближайшего соседа (исключая себя)
            distances.append(sorted_dist[1])
    
    if not distances:
        return 0.6  # Значение по умолчанию
    
    # 75-й перцентиль для строгой кластеризации
    epsilon = np.percentile(distances, 75)
    return epsilon

def verify_cluster_similarity(cluster_paths, threshold=0.6):
    """
    Проверяет, что все лица в кластере действительно принадлежат одному человеку.
    Использует попарное сравнение через face_recognition.
    
    Args:
        cluster_paths: Список путей к изображениям в кластере
        threshold: Порог расстояния (меньше = строже)
    
    Returns:
        True если все лица похожи, False если есть разные люди
    """
    if len(cluster_paths) < 2:
        return True
    
    # Извлекаем эмбеддинги для всех фото в кластере
    embeddings = []
    valid_paths = []
    
    for path in cluster_paths:
        try:
            img = face_recognition.load_image_file(str(path))
            face_encodings = face_recognition.face_encodings(img, model="large")
            
            if face_encodings:
                embeddings.append(face_encodings[0])
                valid_paths.append(path)
        except:
            continue
    
    if len(embeddings) < 2:
        return True
    
    # Попарное сравнение всех лиц
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            distance = np.linalg.norm(embeddings[i] - embeddings[j])
            
            # Если расстояние больше порога - разные люди
            if distance > threshold:
                print(f"  ❌ Обнаружены разные люди: {valid_paths[i].name} и {valid_paths[j].name} (distance: {distance:.3f})")
                return False
    
    return True

def build_plan_live(
    input_dir: Path,
    det_size=(640, 640),
    min_score: float = 0.95,  # Повышен для лучшей точности
    min_cluster_size: int = 1,  # Разрешить одиночные фото
    min_samples: int = 1,  # Разрешить одиночные кластеры
    providers: List[str] = ("CPUExecutionProvider",),  # Для обратной совместимости
    progress_callback=None,
    include_excluded: bool = False,
):
    print(f"🔍 [CLUSTER] build_plan_live вызвана: input_dir={input_dir}, include_excluded={include_excluded}")
    
    try:
        input_dir = Path(input_dir)
        print(f"🔍 [CLUSTER] input_dir преобразован в Path: {input_dir}")
        
        # Собираем все изображения, учитываем флаг include_excluded
        excluded_names = ["общие", "общая", "common", "shared", "все", "all", "mixed", "смешанные"]
        print(f"🔍 [CLUSTER] excluded_names: {excluded_names}")
    except Exception as e:
        print(f"❌ [CLUSTER] Ошибка в начале build_plan_live: {e}")
        raise
    
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
                continue
            
            # Извлекаем эмбеддинги (модель "large" для максимальной точности)
            face_encodings = face_recognition.face_encodings(
                img, 
                known_face_locations=face_locations,
                model="large"  # 128-мерный вектор, 99.38% точность на LFW
            )
            
            if not face_encodings:
                no_faces.append(p)
                continue

            count = 0
            for emb in face_encodings:
                # Эмбеддинги уже нормализованы в face_recognition
                embeddings.append(emb.astype(np.float64))
                owners.append(p)
                count += 1
                processed_faces += 1

            if count > 0:
                img_face_count[p] = count
                
        except Exception as e:
            print(f"  ❌ Ошибка обработки {p.name}: {e}")
            unreadable.append(p)
            continue

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

    # Этап 3: Двухэтапная верификация кластеров
    if progress_callback:
        progress_callback("🔄 Верификация кластеров (2-этапная проверка)...", 90)
    
    print(f"🔍 [CLUSTER] Начинаем верификацию {len(cluster_map)} кластеров")
    
    validated_clusters = {}
    next_id = next_single_label
    
    for cluster_id, paths in cluster_map.items():
        paths_list = list(paths)
        
        # Проверяем, что все лица в кластере действительно одного человека
        if verify_cluster_similarity(paths_list, threshold=0.6):
            validated_clusters[cluster_id] = paths
            print(f"  ✅ Кластер {cluster_id} валиден ({len(paths)} фото)")
        else:
            # Разбить кластер на отдельные папки для каждого фото
            print(f"  ⚠️ Кластер {cluster_id} содержит разных людей, разбиваем на {len(paths)} папок")
            for path in paths:
                validated_clusters[next_id] = {path}
                print(f"    📁 Создан кластер {next_id} для {path.name}")
                next_id += 1
    
    # Обновляем cluster_map и cluster_by_img
    cluster_map = validated_clusters
    cluster_by_img = defaultdict(set)
    for cluster_id, paths in cluster_map.items():
        for path in paths:
            cluster_by_img[path].add(cluster_id)
    
    print(f"🔍 [CLUSTER] После верификации: {len(cluster_map)} валидных кластеров")
    
    # Этап 4: Формирование плана распределения
    if progress_callback:
        progress_callback("🔄 Формирование плана распределения...", 95)
    
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

    # Завершение
    if progress_callback:
        progress_callback(f"✅ Кластеризация завершена! Найдено {len(cluster_map)} кластеров, обработано {len(plan)} изображений", 100)

    print(f"✅ Кластеризация завершена: {input_dir} → кластеров: {len(cluster_map)}, изображений: {len(plan)}")

    return {
        "clusters": {
            int(k): [str(p) for p in sorted(v, key=lambda x: str(x))]
            for k, v in cluster_map.items()
        },
        "plan": plan,
        "unreadable": [str(p) for p in unreadable],
        "no_faces": [str(p) for p in no_faces],
    }

def distribute_to_folders(plan: dict, base_dir: Path, cluster_start: int = 1, progress_callback=None) -> Tuple[int, int, int]:
    moved, copied = 0, 0
    moved_paths = set()

    used_clusters = sorted({c for item in plan.get("plan", []) for c in item["cluster"]})
    cluster_id_map = {old: cluster_start + idx for idx, old in enumerate(used_clusters)}

    plan_items = plan.get("plan", [])
    total_items = len(plan_items)
    
    # Подсчитываем количество файлов в каждом кластере
    cluster_file_counts = {}
    for item in plan_items:
        clusters = [cluster_id_map[c] for c in item["cluster"]]
        for cluster_id in clusters:
            cluster_file_counts[cluster_id] = cluster_file_counts.get(cluster_id, 0) + 1
    
    if progress_callback:
        progress_callback(f"🔄 Распределение {total_items} файлов по папкам...", 0)

    for i, item in enumerate(plan_items):
        if progress_callback:
            percent = int((i + 1) / max(total_items, 1) * 100)
            progress_callback(f"📁 Распределение файлов: {percent}% ({i+1}/{total_items})", percent)
            
        src = Path(item["path"])
        clusters = [cluster_id_map[c] for c in item["cluster"]]
        if not src.exists():
            continue

        if len(clusters) == 1:
            cluster_id = clusters[0]
            dst = base_dir / f"{cluster_id}" / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
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
            except Exception as e:
                print(f"❌ Ошибка перемещения {src} → {dst}: {e}")
        else:
            for cluster_id in clusters:
                dst = base_dir / f"{cluster_id}" / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    # Skip if source and destination are the same file
                    try:
                        if src.resolve() == dst.resolve():
                            print(f"⚠️ Пропуск копирования (одинаковые пути): {src} → {dst}")
                            continue
                    except Exception:
                        if str(src) == str(dst):
                            print(f"⚠️ Пропуск копирования (одинаковые строки): {src} → {dst}")
                            continue
                    shutil.copy2(str(src), str(dst))
                    copied += 1
                except Exception as e:
                    print(f"❌ Ошибка копирования {src} → {dst}: {e}")
            try:
                src.unlink()  # удаляем оригинал после копирования в несколько папок
            except Exception as e:
                print(f"❌ Ошибка удаления {src}: {e}")

    # Переименование папок с указанием количества файлов
    if progress_callback:
        progress_callback("📝 Переименование папок с количеством файлов...", 95)
    
    for cluster_id, file_count in cluster_file_counts.items():
        old_folder = base_dir / str(cluster_id)
        new_folder = base_dir / f"{cluster_id} ({file_count})"
        
        if old_folder.exists() and old_folder.is_dir():
            try:
                old_folder.rename(new_folder)
                print(f"📁 Переименовано: {old_folder.name} → {new_folder.name}")
            except Exception as e:
                print(f"❌ Ошибка переименования {old_folder} → {new_folder}: {e}")

    # Очистка пустых папок
    if progress_callback:
        progress_callback("🧹 Очистка пустых папок...", 100)

    for p in sorted(moved_paths, key=lambda x: len(str(x)), reverse=True):
        try:
            if p.exists() and not any(p.iterdir()):
                p.rmdir()
        except Exception:
            pass

    print(f"📦 Перемещено: {moved}, скопировано: {copied}")
    return moved, copied, cluster_start + len(used_clusters)

def find_common_folders_recursive(root_dir: Path):
    """Рекурсивно найти все папки 'общие' в дереве каталогов"""
    excluded_names = ["общие", "общая", "common", "shared", "все", "all", "mixed", "смешанные"]
    common_folders = []
    
    print(f"🔍 Начинаем поиск папок 'общие' в: {root_dir}")
    print(f"🔍 Ищем папки с именами: {excluded_names}")
    
    def scan_directory(dir_path, level=0):
        indent = "  " * level
        try:
            print(f"{indent}📁 Сканируем: {dir_path}")
            for item in dir_path.iterdir():
                if item.is_dir():
                    print(f"{indent}  🔍 Проверяем папку: {item.name}")
                    # Проверяем, является ли эта папка "общей"
                    item_name_lower = item.name.lower()
                    for ex in excluded_names:
                        if ex in item_name_lower:
                            common_folders.append(item)
                            print(f"{indent}  ✅ Найдена папка 'общие': {item}")
                            break
                    else:
                        # Рекурсивно сканируем подпапки (только до уровня 3)
                        if level < 3:
                            scan_directory(item, level + 1)
        except PermissionError:
            print(f"{indent}❌ Нет доступа к папке: {dir_path}")
        except Exception as e:
            print(f"{indent}❌ Ошибка сканирования {dir_path}: {e}")
    
    scan_directory(root_dir)
    print(f"🔍 Поиск завершен. Найдено {len(common_folders)} папок 'общие': {[str(f) for f in common_folders]}")
    return common_folders


def process_common_folder_at_level(common_dir: Path, progress_callback=None):
    """
    Обработать одну папку 'общие':
    1. Найти всех уникальных людей на фото
    2. Создать папки для каждого человека (даже если он есть только на общих фото)
    3. Создать 2 дополнительные пустые папки
    4. НЕ трогать сами общие фотографии - они остаются в папке 'общие'
    """
    parent_dir = common_dir.parent
    
    print(f"🔍 Обрабатываем папку 'общие': {common_dir}")
    print(f"🔍 Родительская папка: {parent_dir}")
    
    # Кластеризуем ТОЛЬКО фото из папки "общие" чтобы найти всех людей
    print(f"🔍 Вызываем build_plan_live для: {common_dir}")
    data = build_plan_live(common_dir, include_excluded=True, progress_callback=progress_callback)
    plan = data.get('plan', [])
    
    print(f"🔍 Получен план с {len(plan)} файлами")
    if plan:
        print(f"🔍 Первые файлы в плане: {[item['path'] for item in plan[:3]]}")
    
    if not plan:
        print(f"❌ Нет фото для обработки в {common_dir}")
        return 0
    
    # Получаем все существующие ID папок в родительской директории
    existing_ids = set()
    for d in parent_dir.iterdir():
        if d.is_dir():
            try:
                # Пытаемся извлечь число из начала имени папки
                id_str = d.name.split(' ')[0].split('-')[0].split('_')[0]
                if id_str.isdigit():
                    existing_ids.add(int(id_str))
            except:
                continue
    
    print(f"🔍 Существующие ID папок: {sorted(existing_ids)}")
    
    # Собираем все уникальные ID кластеров (людей) из общих фото
    cluster_ids = set()
    for item in plan:
        for cid in item['cluster']:
            cluster_ids.add(cid)
    
    print(f"🔍 Найдено уникальных людей на общих фото: {len(cluster_ids)}")
    print(f"🔍 ID людей: {sorted(cluster_ids)}")
    
    created = 0
    
    # Создаём папки для каждого человека, которого нет в существующих папках
    for cluster_id in sorted(cluster_ids):
        if cluster_id not in existing_ids:
            folder = parent_dir / str(cluster_id)
            folder.mkdir(parents=True, exist_ok=True)
            print(f"📁 Создана папка для человека {cluster_id}: {folder}")
            created += 1
            existing_ids.add(cluster_id)
        else:
            print(f"⏩ Папка для человека {cluster_id} уже существует")
    
    # Объединяем все ID (существующие + созданные)
    all_ids = existing_ids.union(cluster_ids)
    
    # Создаём 2 дополнительные пустые папки с продолжением нумерации
    max_id = max(all_ids) if all_ids else 0
    print(f"🔍 Максимальный ID: {max_id}")
    
    for i in range(1, 3):
        new_id = max_id + i
        folder = parent_dir / str(new_id)
        folder.mkdir(parents=True, exist_ok=True)
        print(f"📁 Создана дополнительная пустая папка {i}/2: {folder}")
        created += 1
    
    print(f"✅ Всего создано папок: {created}")
    print(f"📸 Общие фотографии остались нетронутыми в: {common_dir}")
    
    return created


def process_group_folder(group_dir: Path, progress_callback=None, include_excluded: bool = False):
    """
    Если include_excluded=True, рекурсивно ищем все папки "общие" и копируем фото в папки людей.
    Иначе - обрабатываем каждую подпапку отдельно.
    """
    cluster_counter = 1
    
    import time
    call_id = int(time.time() * 1000) % 10000
    print(f"🔍 process_group_folder [{call_id}] вызвана для: {group_dir}, include_excluded={include_excluded}")
    
    if include_excluded:
        # Рекурсивно находим все папки "общие"
        if progress_callback:
            progress_callback("🔍 Поиск папок 'общие' во всей иерархии...", 10)
        
        common_folders = find_common_folders_recursive(group_dir)
        
        if not common_folders:
            if progress_callback:
                progress_callback("❌ Папки 'общие' не найдены во всей иерархии", 100)
            print(f"❌ Папки 'общие' не найдены в {group_dir}")
            print(f"🔍 Проверили следующие папки:")
            
            def debug_scan_directory(dir_path, level=0):
                indent = "  " * level
                try:
                    print(f"{indent}📁 {dir_path}")
                    for item in dir_path.iterdir():
                        if item.is_dir():
                            print(f"{indent}  └── 📁 {item.name}")
                            if level < 2:  # Ограничиваем глубину
                                debug_scan_directory(item, level + 1)
                except Exception as e:
                    print(f"{indent}  ❌ Ошибка: {e}")
            
            debug_scan_directory(group_dir)
            return 0, 0, cluster_counter
        
        print(f"🔍 Найдено {len(common_folders)} папок 'общие'")
        
        total_copied = 0
        total_folders = len(common_folders)
        
        # Обрабатываем каждую найденную папку "общие"
        for i, common_folder in enumerate(common_folders):
            if progress_callback:
                percent = 20 + int((i + 1) / total_folders * 70)
                progress_callback(f"📋 Обрабатываем папку: {common_folder.name} ({i+1}/{total_folders})", percent)
            
            copied = process_common_folder_at_level(common_folder, progress_callback)
            total_copied += copied
        
        if progress_callback:
            progress_callback(f"✅ Всего скопировано: {total_copied} файлов", 100)
        
        print(f"✅ Обработка общих фото [{call_id}] завершена: скопировано {total_copied} файлов из {len(common_folders)} папок")
        return 0, total_copied, cluster_counter
    # Обрабатываем каждую подпапку, исключая папки 'общие'
    subfolders = [f for f in sorted(group_dir.iterdir()) if f.is_dir() and "общие" not in f.name.lower()]
    total_subfolders = len(subfolders)
    for i, subfolder in enumerate(subfolders):
        if progress_callback:
            percent = 10 + int((i + 1) / max(total_subfolders, 1) * 80)
            progress_callback(f"🔍 Обрабатывается подпапка: {subfolder.name} ({i+1}/{total_subfolders})", percent)
        print(f"🔍 Обрабатывается подпапка [{call_id}]: {subfolder}")
        plan = build_plan_live(subfolder, progress_callback=progress_callback)
        print(f"📊 Кластеров: {len(plan.get('clusters', {}))}, файлов: {len(plan.get('plan', []))}")
        moved, copied, cluster_counter = distribute_to_folders(
            plan, subfolder, cluster_start=cluster_counter, progress_callback=progress_callback
        )




import os
import cv2
import shutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from sklearn.metrics.pairwise import cosine_distances
from insightface.app import FaceAnalysis
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

def build_plan_live(
    input_dir: Path,
    det_size=(640, 640),
    min_score: float = 0.7,
    min_cluster_size: int = 3,
    min_samples: int = 2,
    providers: List[str] = ("CPUExecutionProvider",),
    progress_callback=None,
    include_excluded: bool = False,
):
    input_dir = Path(input_dir)
    # Собираем все изображения, учитываем флаг include_excluded
    excluded_names = ["общие", "общая", "common", "shared", "все", "all", "mixed", "смешанные"]
    
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

    app = FaceAnalysis(name="buffalo_l", providers=list(providers))
    ctx_id = -1 if "cpu" in str(providers).lower() else 0
    app.prepare(ctx_id=ctx_id, det_size=det_size)

    if progress_callback:
        progress_callback("✅ Модель загружена, начинаем анализ изображений...", 10)

    embeddings = []
    owners = []
    img_face_count = {}
    unreadable = []
    no_faces = []

    total = len(all_images)
    processed_faces = 0
    
    for i, p in enumerate(all_images):
        # Обновляем прогресс для каждого изображения
        if progress_callback:
            percent = 10 + int((i + 1) / max(total, 1) * 70)  # 10-80% для анализа изображений
            progress_callback(f"📷 Анализ изображений: {percent}% ({i+1}/{total}) - {p.name}", percent)
        
        img = imread_safe(p)
        if img is None:
            unreadable.append(p)
            continue
            
        faces = app.get(img)
        if not faces:
            no_faces.append(p)
            continue

        count = 0
        for f in faces:
            if getattr(f, "det_score", 1.0) < min_score:
                continue
            emb = getattr(f, "normed_embedding", None)
            if emb is None:
                continue
            emb = emb.astype(np.float64)  # HDBSCAN expects double
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            embeddings.append(emb)
            owners.append(p)
            count += 1
            processed_faces += 1

        if count > 0:
            img_face_count[p] = count

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

    # Этап 2: Кластеризация
    if progress_callback:
        progress_callback(f"🔄 Кластеризация {len(embeddings)} лиц...", 80)
    
    X = np.vstack(embeddings)
    distance_matrix = cosine_distances(X)

    if progress_callback:
        progress_callback("🔄 Вычисление матрицы расстояний...", 85)

    model = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=min_cluster_size, min_samples=min_samples)
    raw_labels = model.fit_predict(distance_matrix)

    # Прямое назначение без merge_clusters_by_centroid для строгой сегрегации
    cluster_map = defaultdict(set)
    cluster_by_img = defaultdict(set)
    for label, path in zip(raw_labels, owners):
        if label == -1:
            continue
        cluster_map[label].add(path)
        cluster_by_img[path].add(label)

    # Этап 3: Формирование плана распределения
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
                    shutil.copy2(str(src), str(dst))
                    copied += 1
                except Exception as e:
                    print(f"❌ Ошибка копирования {src} → {dst}: {e}")
            try:
                src.unlink()  # удаляем оригинал после копирования в несколько папок
            except Exception as e:
                print(f"❌ Ошибка удаления {src}: {e}")

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
    
    def scan_directory(dir_path):
        try:
            for item in dir_path.iterdir():
                if item.is_dir():
                    # Проверяем, является ли эта папка "общей"
                    if any(ex in item.name.lower() for ex in excluded_names):
                        common_folders.append(item)
                        print(f"🔍 Найдена папка 'общие': {item}")
                    else:
                        # Рекурсивно сканируем подпапки
                        scan_directory(item)
        except PermissionError:
            print(f"❌ Нет доступа к папке: {dir_path}")
        except Exception as e:
            print(f"❌ Ошибка сканирования {dir_path}: {e}")
    
    scan_directory(root_dir)
    return common_folders


def process_common_folder_at_level(common_dir: Path, progress_callback=None):
    """Обработать одну папку 'общие' и скопировать фото в папки людей на том же уровне"""
    parent_dir = common_dir.parent
    
    print(f"🔍 Обрабатываем папку 'общие': {common_dir}")
    print(f"🔍 Ищем папки людей в: {parent_dir}")
    
    # Находим папки людей (с номерами кластеров) на том же уровне
    person_dirs = [d for d in parent_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    
    if not person_dirs:
        print(f"❌ Папки людей не найдены в {parent_dir}. Пропускаем.")
        return 0
    
    print(f"🔍 Найдены папки людей: {[d.name for d in person_dirs]}")
    
    # Кластеризуем ТОЛЬКО фото из папки "общие"  
    print(f"🔍 Вызываем build_plan_live для: {common_dir}")
    data = build_plan_live(common_dir, include_excluded=True, progress_callback=progress_callback)
    plan = data.get('plan', [])
    
    print(f"🔍 Получен план с {len(plan)} файлами")
    if plan:
        print(f"🔍 Первые файлы в плане: {[item['path'] for item in plan[:3]]}")
    
    if not plan:
        print(f"❌ Нет фото для обработки в {common_dir}")
        return 0
    
    copied = 0
    
    # Для каждого файла из общей папки копируем в папки кластеров
    for item in plan:
        item_path = Path(item['path'])
        clusters_for_file = item['cluster']
        
        for cluster_id in clusters_for_file:
            # Ищем соответствующую папку человека на том же уровне
            target_dir = parent_dir / str(cluster_id)
            if target_dir.exists():
                dst = target_dir / item_path.name
                
                # Проверяем, что файл не копируется сам в себя
                print(f"🔍 Проверяем копирование: {item_path} → {dst}")
                print(f"🔍 Resolved пути: {item_path.resolve()} vs {dst.resolve()}")
                
                if item_path.resolve() != dst.resolve():
                    try:
                        shutil.copy2(str(item_path), str(dst))
                        copied += 1
                        print(f"📋 Скопировано: {item_path.name} → {target_dir.name}")
                        if progress_callback:
                            progress_callback(f"📋 Копирую {item_path.name} в кластер {cluster_id}", 80)
                    except Exception as e:
                        print(f"❌ Ошибка копирования {item_path} → {dst}: {e}")
                else:
                    print(f"⚠️ Пропускаем копирование файла в себя: {item_path}")
    
    print(f"✅ Папка {common_dir.name}: скопировано {copied} файлов")
    return copied


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




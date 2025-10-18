"""
Production-вариант кластеризации лиц на базе ArcFace + Faiss.
- Детекция и эмбеддинги: InsightFace (ArcFace), app.FaceAnalysis
- Кластеризация: граф по порогу косинусной близости (Faiss range_search + компоненты связности)
- Совместим по интерфейсу с упрощённой версией: build_plan_pro, distribute_to_folders, process_group_folder
- Устойчив к Unicode-путям, много-лицам на фото, копированию для мультикластерных кадров

Зависимости:
    pip install insightface onnxruntime-gpu faiss-gpu opencv-python pillow scikit-learn numpy
или (CPU-only):
    pip install insightface onnxruntime faiss-cpu opencv-python pillow scikit-learn numpy

Автор: prod-ready скелет. Подключайте в своё приложение напрямую.
"""
from __future__ import annotations
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import cv2
from PIL import Image
from collections import defaultdict, deque

# Faiss может отсутствовать при сборке — валидируем импорт.
try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    faiss = None

try:
    from insightface.app import FaceAnalysis
except Exception as e:  # pragma: no cover
    FaceAnalysis = None

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
ProgressCB = Optional[Callable[[str, int], None]]

# ------------------------
# Утилиты ввода/вывода
# ------------------------

def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


def imread_safe(path: Path) -> Optional[np.ndarray]:
    """Аккуратное чтение изображений (BGR->RGB). Возвращает None при ошибке.
    Используем cv2.imdecode для лучшей поддержки Unicode путей.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb
    except Exception:
        return None


# ------------------------
# Инициализация модели InsightFace
# ------------------------
@dataclass
class ArcFaceConfig:
    det_size: Tuple[int, int] = (640, 640)
    ctx_id: int = 0                   # GPU: индекс, CPU: -1
    allowed_blur: float = 0.8         # порог качества (примерный, отфильтруем явный мусор)


class ArcFaceEmbedder:
    def __init__(self, config: ArcFaceConfig = ArcFaceConfig()):
        if FaceAnalysis is None:
            raise ImportError("insightface не установлен. Установите пакет insightface.")
        self.app = FaceAnalysis(name="buffalo_l")
        # ctx_id=-1 → CPU, иначе GPU. det_size влияет на recall/скорость детектора
        self.app.prepare(ctx_id=config.ctx_id, det_size=config.det_size)
        self.allowed_blur = config.allowed_blur

    def extract(self, img_rgb: np.ndarray) -> List[Dict]:
        """Возвращает список лиц: [{embedding, quality, bbox}]. embedding уже L2-нормирован InsightFace."""
        faces = self.app.get(img_rgb)
        results: List[Dict] = []
        for f in faces:
            # f.normed_embedding — L2-нормированный эмбеддинг (512,)
            emb = getattr(f, "normed_embedding", None)
            if emb is None:
                # запасной путь: normalise raw embedding
                raw = getattr(f, "embedding", None)
                if raw is None:
                    continue
                v = np.asarray(raw, dtype=np.float32)
                n = np.linalg.norm(v) + 1e-12
                emb = (v / n).astype(np.float32)
            else:
                emb = np.asarray(emb, dtype=np.float32)

            # эвристика качества: используем blur/pose/детскую confidence если есть
            quality = float(getattr(f, "det_score", 0.99))
            if quality <= 0:  # страховка
                quality = 0.99

            bbox = tuple(int(x) for x in f.bbox.astype(int).tolist())
            results.append({
                "embedding": emb,
                "quality": quality,
                "bbox": bbox,
            })
        return results


# ------------------------
# Кластеризация через Faiss (граф по порогу косинусной близости)
# ------------------------
@dataclass
class ClusterParams:
    sim_threshold: float = 0.60   # чем выше, тем строже (0.55–0.65 — чаще всего ок)
    min_cluster_size: int = 2     # срезаем мелкие компоненты как одиночки
    max_edges_per_node: int = 50  # ограничение на степень узла (ускорение на огромных N)


def _build_similarity_graph_faiss(embeddings: np.ndarray, params: ClusterParams) -> List[List[int]]:
    if faiss is None:
        raise ImportError("faiss не установлен. Установите faiss-gpu или faiss-cpu.")
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    # Эмбеддинги должны быть L2-нормированы для cosine=dot
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    X = embeddings / norms

    d = X.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(X)

    # range_search: вернёт пары (i,j) с sim >= threshold
    lims, D, I = index.range_search(X, params.sim_threshold)

    # Формируем списки смежности (без self-loop и дубликатов)
    n = X.shape[0]
    adj: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        beg, end = lims[i], lims[i + 1]
        # сортируем по sim убыв.
        pairs = sorted(zip(I[beg:end], D[beg:end]), key=lambda t: -t[1])
        count = 0
        for j, sim in pairs:
            if j == i or j < 0:
                continue
            adj[i].append(int(j))
            count += 1
            if params.max_edges_per_node and count >= params.max_edges_per_node:
                break
    return adj


def _connected_components(adj: List[List[int]]) -> np.ndarray:
    n = len(adj)
    labels = -np.ones(n, dtype=np.int32)
    cid = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        # BFS/DFS
        q = deque([i])
        labels[i] = cid
        while q:
            u = q.popleft()
            for v in adj[u]:
                if labels[v] == -1:
                    labels[v] = cid
                    q.append(v)
        cid += 1
    return labels


def cluster_embeddings_faiss(embeddings: np.ndarray, params: ClusterParams) -> np.ndarray:
    if embeddings.size == 0:
        return np.array([], dtype=np.int32)
    adj = _build_similarity_graph_faiss(embeddings, params)
    labels = _connected_components(adj)

    # Отфильтруем мелкие кластеры: одиночки → -1, потом переразметим плотные
    sizes = defaultdict(int)
    for lb in labels:
        sizes[int(lb)] += 1

    for i, lb in enumerate(labels):
        if sizes[int(lb)] < params.min_cluster_size:
            labels[i] = -1

    # Сжимаем индексы кластеров к [0..K-1], игнорируя -1
    uniq = sorted(x for x in set(labels.tolist()) if x != -1)
    remap = {old: i for i, old in enumerate(uniq)}
    out = labels.copy()
    for i, lb in enumerate(labels):
        if lb == -1:
            out[i] = -1
        else:
            out[i] = remap[int(lb)]
    return out


# ------------------------
# Основной пайплайн
# ------------------------

def build_plan_pro(
    input_dir: Path,
    progress_callback: ProgressCB = None,
    sim_threshold: float = 0.60,
    min_cluster_size: int = 2,
    ctx_id: int = 0,
    det_size: Tuple[int, int] = (640, 640),
) -> Dict:
    """Production-кластеризация лиц с ArcFace + Faiss.

    Возвращает dict:
      {
        "clusters": {"0": ["/abs/path/img1.jpg", ...], ...},
        "plan": [ {"path": str, "cluster": [int, ...], "faces": int}, ...],
        "unreadable": [str, ...],
        "no_faces": [str, ...]
      }
    """
    t0 = time.time()
    input_dir = Path(input_dir)
    if progress_callback:
        progress_callback(f"🚀 Кластеризация: {input_dir}", 2)

    # Инициализация эмбеддера
    emb = ArcFaceEmbedder(ArcFaceConfig(det_size=det_size, ctx_id=ctx_id))

    # Сбор изображений
    all_images = [p for p in input_dir.rglob("*") if p.is_file() and is_image(p)]
    if progress_callback:
        progress_callback(f"📂 Найдено изображений: {len(all_images)}", 5)

    owners: List[Path] = []
    all_embeddings: List[np.ndarray] = []
    img_face_count: Dict[Path, int] = {}
    unreadable: List[Path] = []
    no_faces: List[Path] = []

    total = len(all_images)
    for i, img_path in enumerate(all_images):
        if progress_callback and (i % 10 == 0):
            percent = 5 + int((i + 1) / max(1, total) * 60)
            progress_callback(f"📷 Анализ {i+1}/{total}", percent)

        img = imread_safe(img_path)
        if img is None:
            unreadable.append(img_path)
            continue

        faces = emb.extract(img)
        if not faces:
            no_faces.append(img_path)
            continue

        img_face_count[img_path] = len(faces)
        for face in faces:
            all_embeddings.append(face["embedding"])  # уже L2-норм
            owners.append(img_path)

    if not all_embeddings:
        return {
            "clusters": {},
            "plan": [],
            "unreadable": [str(p) for p in unreadable],
            "no_faces": [str(p) for p in no_faces],
        }

    X = np.vstack(all_embeddings).astype(np.float32)

    # Кластеризация через Faiss
    if progress_callback:
        progress_callback("🔗 Построение графа похожести (Faiss)", 70)
    labels = cluster_embeddings_faiss(
        X,
        ClusterParams(sim_threshold=sim_threshold, min_cluster_size=min_cluster_size),
    )

    if progress_callback:
        progress_callback(f"✅ Кластеров: {len(set(labels.tolist()) - {-1})}", 85)

    # Формирование мапов
    cluster_map: Dict[int, set[Path]] = defaultdict(set)
    cluster_by_img: Dict[Path, set[int]] = defaultdict(set)

    for lb, path in zip(labels, owners):
        if lb == -1:
            # одиночки: можно поместить в отдельную папку "-1" либо пропустить из плана
            continue
        cluster_map[int(lb)].add(path)
        cluster_by_img[path].add(int(lb))

    # План перемещений/копирования
    plan: List[Dict] = []
    for path in all_images:
        cl = cluster_by_img.get(path)
        if not cl:
            continue
        plan.append({
            "path": str(path),
            "cluster": sorted(list(cl)),
            "faces": img_face_count.get(path, 0),
        })

    if progress_callback:
        dt = time.time() - t0
        progress_callback(f"⏱️ Обработка завершена за {dt:.1f}с", 95)

    return {
        "clusters": {str(k): [str(p) for p in sorted(v)] for k, v in cluster_map.items()},
        "plan": plan,
        "unreadable": [str(p) for p in unreadable],
        "no_faces": [str(p) for p in no_faces],
    }


# ------------------------
# Распределение по папкам (совместимо с упрощённой версией)
# ------------------------

def distribute_to_folders(plan: dict, base_dir: Path, cluster_start: int = 1, progress_callback: ProgressCB = None, common_mode: bool = False) -> Tuple[int, int, int]:
    import shutil

    moved, copied = 0, 0
    moved_paths = set()

    used_clusters = sorted({c for item in plan.get("plan", []) for c in item["cluster"]})
    # В режиме ОБЩАЯ получаем все кластеры из данных кластеризации
    all_clusters = set()
    if common_mode and "clusters" in plan:
        all_clusters = set(plan["clusters"].keys())
        # Преобразуем строковые ключи в int
        all_clusters = {int(k) for k in all_clusters if k.isdigit()}
        # Объединяем с used_clusters
        used_clusters = sorted(set(used_clusters) | all_clusters)
    
    cluster_id_map = {old: cluster_start + idx for idx, old in enumerate(used_clusters)}

    plan_items = plan.get("plan", [])
    total_items = len(plan_items)
    if progress_callback:
        progress_callback(f"🔄 Распределение {total_items} файлов по папкам...", 0)

    cluster_file_counts: Dict[int, int] = {}
    for item in plan_items:
        clusters = [cluster_id_map[c] for c in item["cluster"]]
        for cid in clusters:
            cluster_file_counts[cid] = cluster_file_counts.get(cid, 0) + 1

    for i, item in enumerate(plan_items):
        if progress_callback:
            percent = int((i + 1) / max(total_items, 1) * 100)
            progress_callback(f"📁 Распределение файлов: {percent}% ({i+1}/{total_items})", percent)

        src = Path(item["path"])  # исходный файл
        clusters = [cluster_id_map[c] for c in item["cluster"]]
        if not src.exists():
            continue
            
        # Проверяем, является ли файл общим (находится в папке "общие")
        is_common_photo = any(excluded_name in str(src.parent).lower() for excluded_name in EXCLUDED_COMMON_NAMES)
        
        if is_common_photo:
            # Общие фотографии НЕ перемещаем - оставляем на месте
            print(f"📌 Общая фотография оставлена: {src.name}")
            # Создаем папки для людей с общих фотографий (пустые) только в режиме ОБЩАЯ
            if common_mode:
                for cid in clusters:
                    empty_folder = base_dir / str(cid)
                    empty_folder.mkdir(parents=True, exist_ok=True)
                    print(f"📁 Создана пустая папка для человека с общих фото: {cid}")
            continue

        if len(clusters) == 1:
            dst = base_dir / f"{clusters[0]}" / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.resolve() != dst.resolve():
                shutil.move(str(src), str(dst))
                moved += 1
                moved_paths.add(src.parent)
        else:
            for cid in clusters:
                dst = base_dir / f"{cid}" / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                if src.resolve() != dst.resolve():
                    shutil.copy2(str(src), str(dst))
                    copied += 1
            try:
                src.unlink()
            except Exception:
                pass

    # Переименование папок: добавляем количество файлов только для непустых папок
    if progress_callback:
        progress_callback("📝 Переименование папок с количеством файлов...", 95)
    for cid, cnt in cluster_file_counts.items():
        if cnt > 0:  # Только для непустых папок
            old_folder = base_dir / str(cid)
            new_folder = base_dir / f"{cid} ({cnt})"
            if old_folder.exists():
                try:
                    old_folder.rename(new_folder)
                except Exception:
                    pass
        else:
            # Удаляем пустые папки
            empty_folder = base_dir / str(cid)
            if empty_folder.exists():
                try:
                    empty_folder.rmdir()
                    print(f"🗑️ Удалена пустая папка: {empty_folder.name}")
                except Exception:
                    pass

    # Чистим пустые каталоги
    if progress_callback:
        progress_callback("🧹 Очистка пустых папок...", 100)
    for p in sorted(moved_paths, key=lambda x: len(str(x)), reverse=True):
        try:
            p.rmdir()
        except Exception:
            pass

    # В режиме ОБЩАЯ создаем пустые папки для всех найденных кластеров + 2 дополнительные
    if common_mode:
        # Создаем папки для всех найденных кластеров с использованием перенумерации
        for old_cid in used_clusters:
            new_cid = cluster_id_map[old_cid]
            empty_folder = base_dir / str(new_cid)
            empty_folder.mkdir(parents=True, exist_ok=True)
            print(f"📁 Создана пустая папка для кластера: {new_cid}")
        
        # Создаем 2 дополнительные пустые папки
        max_mapped_cluster_id = max(cluster_id_map.values()) if cluster_id_map else cluster_start - 1
        for i in range(1, 3):  # Создаем 2 дополнительные папки
            extra_folder = base_dir / str(max_mapped_cluster_id + i)
            extra_folder.mkdir(parents=True, exist_ok=True)
            print(f"📁 Создана дополнительная пустая папка: {max_mapped_cluster_id + i}")

    return moved, copied, cluster_start + len(used_clusters)


# ------------------------
# Групповая обработка и «общие» папки
# ------------------------

EXCLUDED_COMMON_NAMES = ["общие", "общая", "common", "shared", "все", "all", "mixed", "смешанные"]


def find_common_folders_recursive(group_dir: Path) -> List[Path]:
    common: List[Path] = []
    for subdir in group_dir.rglob("*"):
        if subdir.is_dir() and any(ex in subdir.name.lower() for ex in EXCLUDED_COMMON_NAMES):
            common.append(subdir)
    return common


def process_common_folder_at_level(common_dir: Path, progress_callback: ProgressCB = None,
                                   sim_threshold: float = 0.60, min_cluster_size: int = 2,
                                   ctx_id: int = 0, det_size: Tuple[int, int] = (640, 640)) -> Tuple[int, int]:
    """Обработка «общих» папок: раскладываем лица по подпапкам внутри самой «общей».
    Например: common/ → common/1 (...), common/2 (...)
    Возвращает (moved, copied).
    """
    data = build_plan_pro(common_dir, progress_callback=progress_callback,
                          sim_threshold=sim_threshold, min_cluster_size=min_cluster_size,
                          ctx_id=ctx_id, det_size=det_size)
    moved, copied, _ = distribute_to_folders(data, common_dir, cluster_start=1, progress_callback=progress_callback, common_mode=True)
    return moved, copied


def process_group_folder(group_dir: Path, progress_callback: ProgressCB = None,
                         include_excluded: bool = False,
                         sim_threshold: float = 0.60, min_cluster_size: int = 2,
                         ctx_id: int = 0, det_size: Tuple[int, int] = (640, 640)) -> Tuple[int, int, int]:
    """Обрабатывает группу подпапок: кластеризует каждую подпапку отдельно.

    Если include_excluded=False — папки из EXCLUDED_COMMON_NAMES пропускаются.
    Возвращает (moved_total, copied_total, next_cluster_counter).
    """
    group_dir = Path(group_dir)

    if include_excluded:
        commons = find_common_folders_recursive(group_dir)
        for i, c in enumerate(commons):
            if progress_callback:
                progress_callback(f"📋 Общие: {c.name} ({i+1}/{len(commons)})", 5 + int(i / max(1, len(commons)) * 20))
            process_common_folder_at_level(c, progress_callback=progress_callback,
                                           sim_threshold=sim_threshold, min_cluster_size=min_cluster_size,
                                           ctx_id=ctx_id, det_size=det_size)

    subdirs = [d for d in sorted(group_dir.iterdir()) if d.is_dir()]
    if not include_excluded:
        subdirs = [d for d in subdirs if all(ex not in d.name.lower() for ex in EXCLUDED_COMMON_NAMES)]

    total = len(subdirs)
    moved_all, copied_all = 0, 0
    for i, sub in enumerate(subdirs):
        if progress_callback:
            progress_callback(f"🔍 {sub.name}: кластеризация ({i+1}/{total})", 25 + int(i / max(1, total) * 60))
        data = build_plan_pro(
            input_dir=sub,
            progress_callback=progress_callback,
            sim_threshold=sim_threshold,
            min_cluster_size=min_cluster_size,
            ctx_id=ctx_id,
            det_size=det_size,
        )
        m, c, _ = distribute_to_folders(data, sub, cluster_start=1, progress_callback=progress_callback)
        moved_all += m
        copied_all += c

    return moved_all, copied_all, 1


# ------------------------
# CLI-обвязка (опционально)
# ------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ArcFace+Faiss face clustering")
    parser.add_argument("input", type=str, help="Папка с изображениями или группа папок")
    parser.add_argument("--group", action="store_true", help="Обрабатывать как группу подпапок")
    parser.add_argument("--include-common", action="store_true", help="Обрабатывать папки 'общие' внутри группы")
    parser.add_argument("--sim", type=float, default=0.60, help="Порог косинусной близости [0..1]")
    parser.add_argument("--minsz", type=int, default=2, help="Мин. размер кластера")
    parser.add_argument("--cpu", action="store_true", help="Принудительно CPU (ctx_id=-1)")
    parser.add_argument("--det", type=int, nargs=2, default=[640, 640], help="Размер детектора WxH")

    args = parser.parse_args()

    def cb(msg: str, p: int):
        print(f"[{p:3d}%] {msg}")

    if args.group:
        moved, copied, _ = process_group_folder(
            Path(args.input), progress_callback=cb,
            include_excluded=args.include_common,
            sim_threshold=args.sim, min_cluster_size=args.minsz,
            ctx_id=(-1 if args.cpu else 0), det_size=tuple(args.det),
        )
        print(f"DONE: moved={moved}, copied={copied}")
    else:
        data = build_plan_pro(
            Path(args.input), progress_callback=cb,
            sim_threshold=args.sim, min_cluster_size=args.minsz,
            ctx_id=(-1 if args.cpu else 0), det_size=tuple(args.det),
        )
        m, c, _ = distribute_to_folders(data, Path(args.input), cluster_start=1, progress_callback=cb)
        print(f"DONE: moved={m}, copied={c}")

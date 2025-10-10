import os
import cv2
import shutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict

# Optional libs with graceful fallbacks
try:
    import faiss  # faiss-cpu
    _FAISS_OK = True
except Exception:
    faiss = None
    _FAISS_OK = False

try:
    import networkx as nx
    _NX_OK = True
except Exception:
    nx = None
    _NX_OK = False

from sklearn.neighbors import NearestNeighbors  # fallback for kNN

# Optional insightface with graceful fallback
try:
    from insightface.app import FaceAnalysis
    _INSIGHTFACE_OK = True
except Exception as e:
    print(f"⚠️ InsightFace не доступен: {e}")
    FaceAnalysis = None
    _INSIGHTFACE_OK = False

# -------------------------------
# Config / Defaults
# -------------------------------
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

# Quality gates (консервативные дефолты, ориентированы на максимум precision)
MIN_DET_SCORE = 0.80
MIN_BLUR_VAR = 120.0  # var(Laplacian) – грубая оценка резкости

# Hi-Precision graph params (косинусная симметрия на L2-векторах)
KNN_K = 40
T_STRICT = 0.80   # ребро создаётся если sim >= T_STRICT и соседство взаимное
T_MEMBER = 0.78   # фильтр участника относительно медиоида
T_MERGE = 0.82    # (опционально) порог для слияния кластеров по центроидам

# Универсальная адаптация графа (без глобального порога)
DEGREE_TARGET = (2, 4)   # желаемая средняя степень графа после порога
MUTUAL_RANK   = 5        # взаимный ранг соседа (оба в TOP-5 друг у друга)
MIN_SHARED_NEIGHBORS = 4 # минимум общих соседей (по kNN-спискам)

# Общие имена для exclude/include
EXCLUDED_NAMES = ["общие", "общая", "common", "shared", "все", "all", "mixed", "смешанные"]

# -------------------------------
# Utils
# -------------------------------

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def _win_long(path: Path) -> str:
    p = str(path.resolve())
    if os.name == "nt":
        return "\\\\?\\" + p if not p.startswith("\\\\?\\") else p
    return p


def _safe_path(p: Path) -> str:
    return _win_long(p)


def _safe_move(src: Path, dst: Path):
    shutil.move(_safe_path(src), _safe_path(dst))


def _safe_copy(src: Path, dst: Path):
    shutil.copy2(_safe_path(src), _safe_path(dst))


def imread_safe(path: Path):
    try:
        data = np.fromfile(_win_long(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _parse_cluster_id(name: str) -> Optional[int]:
    import re
    m = re.match(r"^\s*(\d+)", name)
    return int(m.group(1)) if m else None


def _collect_existing_numeric_ids(parent_dir: Path) -> Set[int]:
    ids = set()
    for d in parent_dir.iterdir():
        if d.is_dir():
            cid = _parse_cluster_id(d.name)
            if cid is not None:
                ids.add(cid)
    return ids


def _blur_var(img: np.ndarray) -> float:
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# -------------------------------
# Face embeddings extraction
# -------------------------------

def extract_embeddings(
    image_paths: List[Path],
    providers: List[str] = ("CPUExecutionProvider",),
    det_size=(640, 640),
    min_score: float = MIN_DET_SCORE,
    min_blur_var: float = MIN_BLUR_VAR,
    progress_callback=None,
) -> Tuple[np.ndarray, List[Path], Dict[Path, int], List[Path], List[Path]]:
    """
    Возвращает: (X, owners, img_face_count, unreadable, no_faces)
      - X: np.ndarray [N, D] L2-нормированные эмбеддинги
      - owners: list[Path] длины N (соответствие эмбеддинга → исходный файл)
      - img_face_count: dict[Path] → кол-во используемых лиц из изображения
      - unreadable: список битых/нечитаемых файлов
      - no_faces: список файлов без пригодных лиц
    """
    if not _INSIGHTFACE_OK or FaceAnalysis is None:
        if progress_callback:
            progress_callback("❌ InsightFace не доступен. Установите: pip install insightface", 0)
        raise RuntimeError("InsightFace не доступен. Установите: pip install insightface")
    
    try:
        app = FaceAnalysis(name="buffalo_l", providers=list(providers))
        ctx_id = -1 if "cpu" in str(providers).lower() else 0
        app.prepare(ctx_id=ctx_id, det_size=det_size)
    except Exception as e:
        if progress_callback:
            progress_callback(f"❌ Ошибка инициализации InsightFace: {str(e)}", 0)
        raise RuntimeError(f"Ошибка инициализации InsightFace: {str(e)}")

    if progress_callback:
        progress_callback("✅ Модель загружена, начинаем анализ изображений...", 10)

    embeddings: List[np.ndarray] = []
    owners: List[Path] = []
    img_face_count: Dict[Path, int] = {}
    unreadable: List[Path] = []
    no_faces: List[Path] = []

    total = len(image_paths)

    for i, p in enumerate(image_paths):
        if progress_callback:
            percent = 10 + int((i + 1) / max(total, 1) * 60)  # 10-70
            progress_callback(f"📷 Анализ изображений: {percent}% ({i+1}/{total}) - {p.name}", percent)

        img = imread_safe(p)
        if img is None:
            unreadable.append(p)
            continue

        faces = app.get(img)
        if not faces:
            no_faces.append(p)
            continue

        used = 0
        for f in faces:
            if getattr(f, "det_score", 1.0) < min_score:
                continue
            # Blur check на кропе bbox (грубая оценка)
            try:
                x1, y1, x2, y2 = map(int, f.bbox)
                x1 = max(0, x1); y1 = max(0, y1); x2 = min(img.shape[1], x2); y2 = min(img.shape[0], y2)
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                if _blur_var(crop) < min_blur_var:
                    continue
            except Exception:
                # если bbox странный — пропускаем
                continue

            emb = getattr(f, "normed_embedding", None)
            if emb is None:
                continue
            emb = emb.astype(np.float32)
            # L2 нормализация (на всякий случай)
            n = np.linalg.norm(emb)
            if n <= 0:
                continue
            emb = emb / n

            embeddings.append(emb)
            owners.append(p)
            used += 1

        if used > 0:
            img_face_count[p] = used
        else:
            no_faces.append(p)

    if not embeddings:
        return np.empty((0, 512), dtype=np.float32), owners, img_face_count, unreadable, no_faces

    X = np.vstack(embeddings).astype(np.float32)
    # Нормировка для FAISS dot==cos
    if _FAISS_OK:
        try:
            faiss.normalize_L2(X)
        except Exception as e:
            print(f"⚠️ Ошибка FAISS нормализации: {e}")
    return X, owners, img_face_count, unreadable, no_faces


# -------------------------------
# kNN Graph + Mutual Edges + Components
# -------------------------------

def _knn_faiss(X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    d = X.shape[1]
    index = faiss.IndexHNSWFlat(d, 32)
    index.hnsw.efConstruction = 200
    index.add(X)
    index.hnsw.efSearch = 128
    D, I = index.search(X, k)
    # D — squared L2 при IndexHNSWFlat; но после normalize_L2 dot можно получить как 2-0.5*D? Нет. Для надёжности пересчитаем sim как dot.
    # Пересчёт sim будем делать отдельно по индексам I.
    return D, I


def _knn_sklearn_cosine(X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    # cosine distance → 1 - cosine similarity
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='cosine')
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)
    # Превратим в similarity
    sims = 1.0 - distances
    return sims, indices


def build_mutual_edges(
    X: np.ndarray,
    k: int = KNN_K,
    t_strict: Optional[float] = None,  # игнорируется: используем адаптивный порог
    mutual_rank: int = MUTUAL_RANK,
    min_shared_neighbors: int = MIN_SHARED_NEIGHBORS,
) -> List[Tuple[int, int, float]]:
    """Строим список надёжных рёбер с адаптивным порогом.
    1) kNN (FAISS/Sklearn)
    2) локальная robust z-нормализация подобия
    3) усиление плотностью (Jaccard общих соседей)
    4) автопорог по целевой средней степени DEGREE_TARGET
    """
    N = X.shape[0]
    k = min(k, max(2, N))

    # 1) kNN
    if _FAISS_OK and N >= 1000:
        _, I = _knn_faiss(X, k)
        sims = np.zeros((N, k), dtype=np.float32)
        for i in range(N):
            sims[i] = np.dot(X[I[i]].astype(np.float32), X[i].astype(np.float32))
    else:
        sims, I = _knn_sklearn_cosine(X, k)

    # 2) локальная robust z-норма (без self-столбца)
    neigh = sims[:, 1:]
    med = np.median(neigh, axis=1, keepdims=True)
    mad = np.median(np.abs(neigh - med), axis=1, keepdims=True)
    # Исправляем размерности для z-нормализации
    # med_full должен иметь ту же размерность, что и sims
    med_full = np.concatenate([np.zeros((N,1),dtype=np.float32), np.tile(med, (1, k-1))], axis=1)
    mad_full = np.concatenate([np.ones((N,1),dtype=np.float32), np.tile(mad, (1, k-1))], axis=1)
    z = (sims - med_full) / (1.4826*(mad_full+1e-6))

    neighbor_sets = [set(I[i, 1:]) for i in range(N)]
    rank = [{int(I[i, r]): r for r in range(1, I.shape[1])} for i in range(N)]

    # 3) кандидаты с симметричным z и Jaccard
    cand = []  # (i, j, w)
    for i in range(N):
        for jpos, j in enumerate(I[i, 1:]):
            if j < 0 or j == i:
                continue
            # взаимный TOP-R
            ri = rank[i].get(j, 10**9)
            rj = rank[j].get(i, 10**9)
            if ri > mutual_rank or rj > mutual_rank:
                continue
            # общие соседи
            inter = len(neighbor_sets[i].intersection(neighbor_sets[j]))
            if inter < min_shared_neighbors:
                continue
            # симметричный z
            zj_i = float(z[i, jpos+1])
            j_rank = rank[j].get(i, None)
            zj_j = float(z[j, j_rank]) if j_rank is not None else 0.0
            zsym = 0.5*(zj_i + zj_j)
            # Jaccard
            jac = inter / max(1, len(neighbor_sets[i].union(neighbor_sets[j])))
            w = zsym * (0.5 + 0.5*jac)
            cand.append((i, j, w))

    if not cand:
        return []

    # Симметризация пар (i<j)
    pair = {}
    for i, j, w in cand:
        a, b = (i, j) if i < j else (j, i)
        pair.setdefault((a, b), []).append(w)
    scores = np.array([np.mean(ws) for ws in pair.values()], dtype=np.float32)

    # 4) автопорог по DEGREE_TARGET
    def avg_degree(th):
        m = 0
        for w in scores:
            if w >= th:
                m += 1
        return (2.0*m) / max(1, N)

    low, high = np.percentile(scores, 10), np.percentile(scores, 90)
    target_lo, target_hi = DEGREE_TARGET
    t = (low + high) / 2.0
    for _ in range(20):
        deg = avg_degree(t)
        if deg < target_lo:
            high = t
        elif deg > target_hi:
            low = t
        else:
            break
        t = (low + high) / 2.0

    edges = []
    idx = 0
    for (a, b), wlist in pair.items():
        score = float(scores[idx]); idx += 1
        if score >= t:
            edges.append((a, b, score))
    return edges


def connected_components_from_edges(N: int, edges: List[Tuple[int, int, float]]) -> List[List[int]]:
    if _NX_OK:
        G = nx.Graph()
        G.add_nodes_from(range(N))
        for u, v, w in edges:
            G.add_edge(u, v, weight=w)
        return [list(c) for c in nx.connected_components(G)]
    # Fallback: union-find (быстро и без зависимостей)
    parent = list(range(N))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for u, v, _ in edges:
        union(u, v)

    groups = defaultdict(list)
    for i in range(N):
        groups[find(i)].append(i)
    return list(groups.values())


# -------------------------------
# Medoid membership filtering & optional inter-cluster merge
# -------------------------------

def _medoid_index(points: np.ndarray) -> int:
    # Выбираем точку с максимальной суммой косинусных сходств к другим
    # эквивалентно argmax по сумме dot, так как X уже L2-нормирован
    S = np.dot(points, points.T)
    sums = S.sum(axis=1)
    return int(np.argmax(sums))


def filter_by_medoid(
    X: np.ndarray,
    components: List[List[int]],
    t_member: float = T_MEMBER,
) -> List[List[int]]:
    filtered = []
    for comp in components:
        if len(comp) == 1:
            filtered.append(comp)
            continue
        P = X[comp]
        m_idx_local = _medoid_index(P)
        m = P[m_idx_local]
        sims = np.dot(P, m)
        keep = [comp[i] for i, s in enumerate(sims) if float(s) >= t_member]
        if len(keep) >= 1:
            filtered.append(keep)
    return filtered


def optional_merge_by_centroids(
    X: np.ndarray,
    clusters: List[List[int]],
    t_merge: float = T_MERGE,
    cluster_attrs: Optional[List[dict]] = None,
) -> List[List[int]]:
    # По умолчанию для максимума precision этот этап не активируем (оставляем как утилиту)
    C = [np.mean(X[c], axis=0) for c in clusters]
    C = [c / (np.linalg.norm(c) + 1e-12) for c in C]

    merged = []
    used = [False] * len(clusters)

    def attrs_conflict(a, b) -> bool:
        if not cluster_attrs:
            return False
        A, B = cluster_attrs[a], cluster_attrs[b]
        # Простейшие гейты: если оба знают gender и они разные — конфликт
        ga, gb = A.get('gender'), B.get('gender')
        if ga is not None and gb is not None and ga != gb:
            return True
        # Возрастные бины: если отличаются сильно — блок
        ya, yb = A.get('age'), B.get('age')
        if ya is not None and yb is not None and abs(ya - yb) >= 15:
            return True
        return False

    for i in range(len(clusters)):
        if used[i]:
            continue
        cur = list(clusters[i])
        used[i] = True
        for j in range(i + 1, len(clusters)):
            if used[j]:
                continue
            sim = float(np.dot(C[i], C[j]))
            if sim >= t_merge and not attrs_conflict(i, j):
                cur.extend(clusters[j])
                used[j] = True
        merged.append(sorted(cur))
    return merged


# -------------------------------
# High-Precision clustering (end-to-end)
# -------------------------------

def hi_precision_cluster(
    X: np.ndarray,
    t_member: float = T_MEMBER,
    allow_merge: bool = False,
    t_merge: float = T_MERGE,
    progress_callback=None,
) -> List[List[int]]:
    N = X.shape[0]
    if N == 0:
        return []

    if progress_callback:
        progress_callback("🔗 Строим kNN-граф (адаптивный порог)...", 75)
    edges = build_mutual_edges(X, k=KNN_K)

    if progress_callback:
        progress_callback("🧩 Связные компоненты...", 82)
    comps = connected_components_from_edges(N, edges)

    if progress_callback:
        progress_callback("🎯 Фильтрация по медиоиду...", 88)
    clusters = filter_by_medoid(X, comps, t_member=t_member)

    if allow_merge:
        if progress_callback:
            progress_callback("🧬 Опциональное слияние по центроидам...", 92)
        clusters = optional_merge_by_centroids(X, clusters, t_merge=t_merge, cluster_attrs=None)

    return clusters

# -------------------------------
# Public API: build_plan_live (Hi-Precision)
# -------------------------------

def build_plan_live(
    input_dir: Path,
    det_size=(640, 640),
    min_score: float = MIN_DET_SCORE,
    min_cluster_size: int = 2,  # not used in graph flow; оставлено для совместимости сигнатуры
    min_samples: int = 2,       # not used
    providers: List[str] = ("CPUExecutionProvider",),
    progress_callback=None,
    include_excluded: bool = False,
    allow_merge: bool = False,
    t_strict: float = T_STRICT,
    t_member: float = T_MEMBER,
    t_merge: float = T_MERGE,
):
    """
    Hi-Precision версия build_plan_live: без O(N^2), с графовой кластеризацией и медиоидной фильтрацией.
    Возвращает словарь с ключами: clusters, plan, unreadable, no_faces.
    """
    input_dir = Path(input_dir)
    if include_excluded:
        all_images = [p for p in input_dir.rglob('*') if is_image(p)]
    else:
        all_images = [p for p in input_dir.rglob('*') if is_image(p) and not any(ex in str(p).lower() for ex in EXCLUDED_NAMES)]

    if progress_callback:
        progress_callback(f"📂 Сканируется: {input_dir}, найдено изображений: {len(all_images)}", 1)

    # 1) Embeddings
    try:
        X, owners, img_face_count, unreadable, no_faces = extract_embeddings(
            all_images,
            providers=providers,
            det_size=det_size,
            min_score=min_score,
            progress_callback=progress_callback,
        )
    except Exception as e:
        if progress_callback:
            progress_callback(f"❌ Ошибка извлечения эмбеддингов: {str(e)}", 100)
        return {
            "clusters": {},
            "plan": [],
            "unreadable": [str(p) for p in all_images],
            "no_faces": [],
            "error": str(e)
        }

    if X.shape[0] == 0:
        if progress_callback:
            progress_callback("⚠️ Не найдено лиц для кластеризации", 100)
        return {
            "clusters": {},
            "plan": [],
            "unreadable": [str(p) for p in unreadable],
            "no_faces": [str(p) for p in no_faces],
        }

    # 2) Graph clustering
    if progress_callback:
        progress_callback(f"🔄 Графовая кластеризация {X.shape[0]} лиц...", 80)
    clusters_idx = hi_precision_cluster(
        X,
        t_member=t_member,
        allow_merge=allow_merge,
        t_merge=t_merge,
        progress_callback=progress_callback,
    )

    # 3) Сформировать карты
    cluster_map: Dict[int, Set[Path]] = defaultdict(set)
    cluster_by_img: Dict[Path, Set[int]] = defaultdict(set)
    for new_label, comp in enumerate(clusters_idx):
        for idx in comp:
            p = owners[idx]
            cluster_map[new_label].add(p)
            cluster_by_img[p].add(new_label)

    # 4) План распределения
    if progress_callback:
        progress_callback("📦 Формирование плана распределения...", 95)
    plan = []
    # Обходим все изображения из выборки, чтобы сохранить порядок и учесть multi-face
    seen = set()
    for p in owners:
        if p in seen:
            continue
        seen.add(p)
        clusters = sorted(list(cluster_by_img.get(p, set())))
        if clusters:
            plan.append({
                "path": str(p),
                "cluster": clusters,
                "faces": img_face_count.get(p, 0)
            })

    if progress_callback:
        progress_callback(f"✅ Кластеризация завершена! Найдено {len(cluster_map)} кластеров, обработано {len(plan)} изображений", 100)

    return {
        "clusters": {int(k): [str(x) for x in sorted(v, key=lambda s: str(s))] for k, v in cluster_map.items()},
        "plan": plan,
        "unreadable": [str(p) for p in unreadable],
        "no_faces": [str(p) for p in no_faces],
    }


# -------------------------------
# Distribution (фикс внутренней логики)
# -------------------------------

def distribute_to_folders(
    plan: dict,
    base_dir: Path,
    cluster_start: int = 1,
    progress_callback=None,
    keep_original_on_multi: bool = True,
    annotate_folders: bool = False
) -> Tuple[int, int, int]:
    moved, copied = 0, 0
    moved_paths = set()

    used_clusters = sorted({c for item in plan.get("plan", []) for c in item["cluster"]})
    cluster_id_map = {old: cluster_start + idx for idx, old in enumerate(used_clusters)}

    plan_items = plan.get("plan", [])
    total_items = len(plan_items)

    cluster_file_counts: Dict[int, int] = {}
    for item in plan_items:
        for cluster_id in (cluster_id_map[c] for c in item["cluster"]):
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
                try:
                    if src.resolve() == dst.resolve():
                        continue
                except Exception:
                    if str(src) == str(dst):
                        continue
                _safe_move(src, dst)
                moved += 1
                moved_paths.add(src.parent)
            except Exception as e:
                print(f"❌ Ошибка перемещения {src} → {dst}: {e}")
        else:
            # multi: копируем в каждый кластер, оригинал оставляем
            for cluster_id in clusters:
                dst = base_dir / f"{cluster_id}" / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    try:
                        if src.resolve() == dst.resolve():
                            continue
                    except Exception:
                        if str(src) == str(dst):
                            continue
                    _safe_copy(src, dst)
                    copied += 1
                except Exception as e:
                    print(f"❌ Ошибка копирования {src} → {dst}: {e}")

    # Переименование папок по количеству делаем опциональным (по умолчанию — Off)
    if annotate_folders and cluster_file_counts:
        if progress_callback:
            progress_callback("📝 Обновление заголовков папок...", 95)
        for cluster_id, file_count in cluster_file_counts.items():
            old_folder = base_dir / str(cluster_id)
            new_folder = base_dir / f"{cluster_id} ({file_count})"
            if old_folder.exists() and old_folder.is_dir():
                try:
                    old_folder.rename(new_folder)
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

    return moved, copied, cluster_start + len(used_clusters)


# -------------------------------
# "Общие" папки: поиск и обработка
# -------------------------------

def find_common_folders_recursive(root_dir: Path, max_depth: int = 3):
    common_folders = []

    def scan_directory(dir_path: Path, level=0):
        if level > max_depth:
            return
        try:
            for item in dir_path.iterdir():
                if item.is_dir():
                    low = item.name.lower()
                    if any(ex in low for ex in EXCLUDED_NAMES):
                        common_folders.append(item)
                    else:
                        scan_directory(item, level + 1)
        except PermissionError:
            print(f"❌ Нет доступа к папке: {dir_path}")
        except Exception as e:
            print(f"❌ Ошибка сканирования {dir_path}: {e}")

    scan_directory(Path(root_dir))
    return common_folders


def process_common_folder_at_level(common_dir: Path, progress_callback=None) -> int:
    parent_dir = common_dir.parent
    existing_ids = _collect_existing_numeric_ids(parent_dir)

    data = build_plan_live(common_dir, include_excluded=True, progress_callback=progress_callback)
    plan = data.get('plan', [])
    if not plan:
        return 0

    # Маппинг: сохраняем существующие, остальным даём новые ID последовательно
    used_clusters_src = sorted({c for item in plan for c in item['cluster']})
    next_id = (max(existing_ids) + 1) if existing_ids else 1
    cluster_id_map = {}
    for old in used_clusters_src:
        if old in existing_ids:
            cluster_id_map[old] = old
        else:
            cluster_id_map[old] = next_id
            next_id += 1

    remapped_plan = []
    for item in plan:
        remapped_plan.append({
            'path': item['path'],
            'cluster': sorted(cluster_id_map[c] for c in item['cluster']),
            'faces': item.get('faces', 0)
        })

    moved, copied, _ = distribute_to_folders(
        {"plan": remapped_plan},
        base_dir=parent_dir,
        cluster_start=1,
        progress_callback=progress_callback,
        keep_original_on_multi=True,
        annotate_folders=False,
    )
    # Возвращаем кол-во копий как показатель работы
    return copied


def process_group_folder(group_dir: Path, progress_callback=None, include_excluded: bool = False):
    cluster_counter = 1
    group_dir = Path(group_dir)

    if include_excluded:
        if progress_callback:
            progress_callback("🔍 Поиск папок 'общие' во всей иерархии...", 10)
        common_folders = find_common_folders_recursive(group_dir)
        if not common_folders:
            if progress_callback:
                progress_callback("❌ Папки 'общие' не найдены", 100)
            return 0, 0, cluster_counter

        total_copied = 0
        total = len(common_folders)
        for i, common_folder in enumerate(common_folders):
            if progress_callback:
                percent = 20 + int((i + 1) / total * 70)
                progress_callback(f"📋 Обрабатывается: {common_folder.name} ({i+1}/{total})", percent)
            total_copied += process_common_folder_at_level(common_folder, progress_callback)

        if progress_callback:
            progress_callback(f"✅ Скопировано из 'общие': {total_copied} файлов", 100)
        return 0, total_copied, cluster_counter

    # Иначе — обрабатываем каждую подпапку отдельно, исключая 'общие'
    subfolders = [f for f in sorted(group_dir.iterdir()) if f.is_dir() and "общие" not in f.name.lower()]
    total_subfolders = len(subfolders)

    for i, subfolder in enumerate(subfolders):
        if progress_callback:
            percent = 10 + int((i + 1) / max(total_subfolders, 1) * 80)
            progress_callback(f"🔍 Обрабатывается подпапка: {subfolder.name} ({i+1}/{total_subfolders})", percent)

        plan = build_plan_live(subfolder, progress_callback=progress_callback)
        moved, copied, cluster_counter = distribute_to_folders(
            plan, subfolder, cluster_start=cluster_counter, progress_callback=progress_callback
        )

    return 0, 0, cluster_counter

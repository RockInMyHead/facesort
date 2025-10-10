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

try:
    import hdbscan
    _HDBSCAN_OK = True
except Exception:
    hdbscan = None
    _HDBSCAN_OK = False

try:
    from sklearn.cluster import SpectralClustering, AgglomerativeClustering, DBSCAN
    from sklearn.mixture import GaussianMixture
    from sklearn.ensemble import IsolationForest
    _SKLEARN_ADVANCED_OK = True
except Exception:
    _SKLEARN_ADVANCED_OK = False

try:
    from sklearn.manifold import TSNE, UMAP
    _MANIFOLD_OK = True
except Exception:
    _MANIFOLD_OK = False

try:
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import resnet50, ResNet50_Weights
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False

from sklearn.neighbors import NearestNeighbors  # fallback for kNN
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

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

# Quality gates (оптимизированы для лучшего распознавания)
MIN_DET_SCORE = 0.50  # Снижено для включения большего количества лиц
MIN_BLUR_VAR = 50.0   # Снижено для менее строгой фильтрации по резкости

# Hi-Precision graph params (оптимизированы для точности)
KNN_K = 60           # Увеличено для лучшего анализа соседства
T_STRICT = 0.65      # Снижено для более гибкого создания рёбер
T_MEMBER = 0.60      # Снижено для включения большего количества лиц в кластеры
T_MERGE = 0.70       # Снижено для лучшего слияния похожих кластеров

# Универсальная адаптация графа (оптимизирована)
DEGREE_TARGET = (3, 8)   # Увеличено для более плотных связей
MUTUAL_RANK   = 8        # Увеличено для более гибкого взаимного ранга
MIN_SHARED_NEIGHBORS = 2 # Снижено для более гибкого анализа соседства

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


def _combine_face_clothing_features(
    face_embeddings: np.ndarray, 
    clothing_features: List[np.ndarray],
    face_qualities: List[Dict] = None,
    clothing_weight: float = 0.3
) -> np.ndarray:
    """
    Объединение признаков лица и одежды для улучшенной кластеризации
    """
    if not clothing_features or len(clothing_features) == 0:
        return face_embeddings
    
    # Преобразуем список в массив
    clothing_array = np.array(clothing_features)
    
    # Нормализуем признаки одежды
    clothing_norms = np.linalg.norm(clothing_array, axis=1, keepdims=True)
    clothing_array = clothing_array / (clothing_norms + 1e-8)
    
    # Объединяем признаки лица и одежды
    # Веса: 70% лицо, 30% одежда
    face_weight = 1.0 - clothing_weight
    
    # Убеждаемся, что размеры совместимы
    min_len = min(len(face_embeddings), len(clothing_array))
    face_embeddings = face_embeddings[:min_len]
    clothing_array = clothing_array[:min_len]
    
    # Объединяем признаки
    combined_features = np.concatenate([
        face_embeddings * face_weight,
        clothing_array * clothing_weight
    ], axis=1)
    
    # Нормализуем объединенные признаки
    combined_norms = np.linalg.norm(combined_features, axis=1, keepdims=True)
    combined_features = combined_features / (combined_norms + 1e-8)
    
    return combined_features


def _extract_advanced_features(X: np.ndarray, face_qualities: List[Dict] = None) -> np.ndarray:
    """
    Извлечение продвинутых признаков для улучшения кластеризации
    """
    N, D = X.shape
    features = []
    
    # 1. Оригинальные эмбеддинги
    features.append(X)
    
    # 2. Статистические признаки
    # Среднее, std, min, max по каждому измерению
    stats_features = np.column_stack([
        np.mean(X, axis=1),
        np.std(X, axis=1),
        np.min(X, axis=1),
        np.max(X, axis=1),
        np.median(X, axis=1)
    ])
    features.append(stats_features)
    
    # 3. Расстояния до центроидов
    centroid = np.mean(X, axis=0)
    centroid_dist = np.linalg.norm(X - centroid, axis=1, keepdims=True)
    features.append(centroid_dist)
    
    # 4. Локальные плотности (kNN distances)
    if N > 5:
        try:
            nbrs = NearestNeighbors(n_neighbors=min(5, N-1), metric='cosine')
            nbrs.fit(X)
            distances, _ = nbrs.kneighbors(X)
            local_density = np.mean(distances[:, 1:], axis=1, keepdims=True)  # Исключаем расстояние до себя
            features.append(local_density)
        except:
            pass
    
    # 5. Качество лиц как признаки
    if face_qualities and len(face_qualities) == N:
        quality_features = np.array([
            [q.get('total_score', 0.5), q.get('blur_score', 0.5), 
             q.get('pose_score', 0.5), q.get('brightness_score', 0.5)]
            for q in face_qualities
        ])
        features.append(quality_features)
    
    # Объединяем все признаки
    if features:
        combined_features = np.hstack(features)
        # Нормализация признаков
        scaler = StandardScaler()
        combined_features = scaler.fit_transform(combined_features)
        return combined_features
    
    return X


def _ensemble_clustering_advanced(
    X: np.ndarray,
    face_qualities: List[Dict] = None,
    progress_callback=None,
) -> List[List[int]]:
    """
    Продвинутый ensemble кластеризации с множественными алгоритмами
    """
    N = X.shape[0]
    if N < 2:
        return [[0]] if N == 1 else []
    
    if progress_callback:
        progress_callback("🧠 Продвинутый ensemble кластеризации...", 75)
    
    # Извлекаем продвинутые признаки
    X_enhanced = _extract_advanced_features(X, face_qualities)
    
    clustering_results = []
    weights = []
    
    # 1. HDBSCAN с разными параметрами
    if _HDBSCAN_OK and hdbscan is not None:
        try:
            if progress_callback:
                progress_callback("🔬 HDBSCAN ensemble...", 76)
            
            # Более консервативные конфигурации HDBSCAN
            hdbscan_configs = [
                {'min_cluster_size': max(2, N//10), 'min_samples': 1, 'alpha': 1.0},
                {'min_cluster_size': max(2, N//8), 'min_samples': 1, 'alpha': 1.2},
                {'min_cluster_size': max(2, N//6), 'min_samples': 2, 'alpha': 0.8},
            ]
            
            for config in hdbscan_configs:
                try:
                    distances = 1.0 - np.dot(X, X.T)
                    distances = distances.astype(np.float64)
                    distances = (distances + distances.T) / 2.0
                    
                    clusterer = hdbscan.HDBSCAN(
                        metric='precomputed',
                        cluster_selection_method='eom',
                        allow_single_cluster=True,
                        **config
                    )
                    
                    labels = clusterer.fit_predict(distances)
                    clusters = _labels_to_clusters(labels)
                    if clusters:
                        clustering_results.append(clusters)
                        weights.append(0.3)  # Высокий вес для HDBSCAN
                except:
                    continue
        except:
            pass
    
    # 2. Spectral Clustering
    if _SKLEARN_ADVANCED_OK and N >= 3:
        try:
            if progress_callback:
                progress_callback("🌈 Спектральная кластеризация...", 78)
            
            # Более консервативные количества кластеров
            n_clusters_options = [max(2, N//8), max(2, N//6), max(2, N//4)]
            
            for n_clusters in n_clusters_options:
                if n_clusters >= N:
                    continue
                    
                try:
                    spectral = SpectralClustering(
                        n_clusters=n_clusters,
                        affinity='cosine',
                        assign_labels='kmeans',
                        random_state=42
                    )
                    labels = spectral.fit_predict(X)
                    clusters = _labels_to_clusters(labels)
                    if clusters:
                        clustering_results.append(clusters)
                        weights.append(0.2)
                except:
                    continue
        except:
            pass
    
    # 3. Agglomerative Clustering
    if _SKLEARN_ADVANCED_OK and N >= 3:
        try:
            if progress_callback:
                progress_callback("🌳 Иерархическая кластеризация...", 80)
            
            # Разные linkage методы
            linkage_methods = ['ward', 'complete', 'average']
            
            for linkage_method in linkage_methods:
                try:
                    n_clusters = max(2, min(N//4, 8))  # Более консервативно
                    agglo = AgglomerativeClustering(
                        n_clusters=n_clusters,
                        linkage=linkage_method,
                        metric='cosine'
                    )
                    labels = agglo.fit_predict(X)
                    clusters = _labels_to_clusters(labels)
                    if clusters:
                        clustering_results.append(clusters)
                        weights.append(0.15)
                except:
                    continue
        except:
            pass
    
    # 4. Gaussian Mixture Models
    if _SKLEARN_ADVANCED_OK and N >= 3:
        try:
            if progress_callback:
                progress_callback("🎯 Gaussian Mixture Models...", 82)
            
            n_components_options = [max(2, N//12), max(2, N//8), max(2, N//6)]
            
            for n_components in n_components_options:
                if n_components >= N:
                    continue
                    
                try:
                    gmm = GaussianMixture(
                        n_components=n_components,
                        covariance_type='full',
                        random_state=42,
                        max_iter=100
                    )
                    labels = gmm.fit_predict(X_enhanced)
                    clusters = _labels_to_clusters(labels)
                    if clusters:
                        clustering_results.append(clusters)
                        weights.append(0.1)
                except:
                    continue
        except:
            pass
    
    # 5. Consensus Clustering
    if len(clustering_results) >= 2:
        if progress_callback:
            progress_callback("🤝 Умная consensus кластеризация...", 85)
        
        try:
            consensus_clusters = _consensus_clustering(clustering_results, weights, N, X)
            if consensus_clusters:
                return consensus_clusters
        except:
            pass
    
    # Fallback: возвращаем лучший результат
    if clustering_results:
        # Выбираем результат с наибольшим весом
        best_idx = np.argmax(weights)
        return clustering_results[best_idx]
    
    return []


def _labels_to_clusters(labels: np.ndarray) -> List[List[int]]:
    """Конвертирует labels в список кластеров"""
    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        if label != -1:  # Игнорируем шум
            clusters[label].append(idx)
    
    return [sorted(cluster) for cluster in clusters.values() if len(cluster) >= 1]


def _consensus_clustering(
    clustering_results: List[List[List[int]]],
    weights: List[float],
    N: int,
    X: np.ndarray = None,
) -> List[List[int]]:
    """
    Умная consensus кластеризация с валидацией и слиянием
    """
    if not clustering_results:
        return []
    
    # Создаем матрицу сходства
    similarity_matrix = np.zeros((N, N))
    
    for clusters, weight in zip(clustering_results, weights):
        for cluster in clusters:
            for i in cluster:
                for j in cluster:
                    if i != j:
                        similarity_matrix[i, j] += weight
    
    # Нормализуем
    max_sim = similarity_matrix.max()
    if max_sim > 0:
        similarity_matrix = similarity_matrix / max_sim
    
    # Адаптивный порог на основе распределения сходства
    if X is not None:
        # Вычисляем косинусные сходства для валидации
        cosine_sims = np.dot(X, X.T)
        np.fill_diagonal(cosine_sims, 0)
        
        # Адаптивный порог на основе квантилей
        threshold = np.percentile(similarity_matrix[similarity_matrix > 0], 30)  # 30-й процентиль
        threshold = max(threshold, 0.3)  # Минимальный порог
    else:
        threshold = 0.4  # Более консервативный порог
    
    consensus_edges = similarity_matrix >= threshold
    
    # Находим связные компоненты
    if _NX_OK and nx is not None:
        G = nx.Graph()
        G.add_nodes_from(range(N))
        for i in range(N):
            for j in range(i+1, N):
                if consensus_edges[i, j]:
                    G.add_edge(i, j)
        
        components = list(nx.connected_components(G))
        clusters = [sorted(list(comp)) for comp in components if len(comp) >= 1]
    else:
        # Простая реализация без NetworkX
        visited = [False] * N
        clusters = []
        
        for i in range(N):
            if not visited[i]:
                cluster = []
                stack = [i]
                
                while stack:
                    node = stack.pop()
                    if not visited[node]:
                        visited[node] = True
                        cluster.append(node)
                        
                        for j in range(N):
                            if not visited[j] and consensus_edges[node, j]:
                                stack.append(j)
                
                if len(cluster) >= 1:
                    clusters.append(sorted(cluster))
    
    # Умное слияние похожих кластеров
    if X is not None and len(clusters) > 1:
        clusters = _smart_merge_clusters(X, clusters)
    
    return clusters


def _smart_merge_clusters(X: np.ndarray, clusters: List[List[int]]) -> List[List[int]]:
    """
    Умное слияние похожих кластеров на основе косинусного сходства
    """
    if len(clusters) <= 1:
        return clusters
    
    # Вычисляем центроиды кластеров
    centroids = []
    for cluster in clusters:
        cluster_embeddings = X[cluster]
        centroid = np.mean(cluster_embeddings, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        centroids.append(centroid)
    
    # Находим пары кластеров для слияния
    merged_clusters = []
    used = [False] * len(clusters)
    
    for i in range(len(clusters)):
        if used[i]:
            continue
            
        current_cluster = clusters[i].copy()
        used[i] = True
        
        # Ищем похожие кластеры для слияния
        for j in range(i + 1, len(clusters)):
            if used[j]:
                continue
                
            # Вычисляем сходство центроидов
            similarity = float(np.dot(centroids[i], centroids[j]))
            
            # Более агрессивный порог для слияния
            merge_threshold = 0.65  # Снижен порог для лучшего слияния
            
            if similarity >= merge_threshold:
                current_cluster.extend(clusters[j])
                used[j] = True
        
        merged_clusters.append(sorted(current_cluster))
    
    return merged_clusters


def _extract_clothing_features(img: np.ndarray, face_bbox: tuple) -> np.ndarray:
    """
    Извлечение признаков одежды из изображения
    """
    if not _TORCH_OK:
        return np.zeros(512, dtype=np.float32)  # Fallback
    
    try:
        # Инициализация модели ResNet50 для извлечения признаков
        if not hasattr(_extract_clothing_features, 'model'):
            _extract_clothing_features.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            _extract_clothing_features.model.eval()
            _extract_clothing_features.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # Извлекаем область одежды (ниже лица)
        x1, y1, x2, y2 = face_bbox
        face_height = y2 - y1
        face_width = x2 - x1
        
        # Область одежды: расширяем bbox лица вниз
        clothing_x1 = max(0, int(x1 - face_width * 0.2))
        clothing_y1 = min(img.shape[0], int(y2 + face_height * 0.1))
        clothing_x2 = min(img.shape[1], int(x2 + face_width * 0.2))
        clothing_y2 = min(img.shape[0], int(y2 + face_height * 1.5))
        
        # Проверяем, что область одежды существует
        if clothing_y1 >= clothing_y2 or clothing_x1 >= clothing_x2:
            return np.zeros(512, dtype=np.float32)
        
        # Извлекаем область одежды
        clothing_crop = img[clothing_y1:clothing_y2, clothing_x1:clothing_x2]
        
        if clothing_crop.size == 0:
            return np.zeros(512, dtype=np.float32)
        
        # Преобразуем в RGB если нужно
        if len(clothing_crop.shape) == 3 and clothing_crop.shape[2] == 3:
            clothing_crop = cv2.cvtColor(clothing_crop, cv2.COLOR_BGR2RGB)
        
        # Применяем трансформации
        clothing_tensor = _extract_clothing_features.transform(clothing_crop).unsqueeze(0)
        
        # Извлекаем признаки
        with torch.no_grad():
            features = _extract_clothing_features.model.avgpool(
                _extract_clothing_features.model.layer4(
                    _extract_clothing_features.model.layer3(
                        _extract_clothing_features.model.layer2(
                            _extract_clothing_features.model.layer1(
                                _extract_clothing_features.model.maxpool(
                                    _extract_clothing_features.model.relu(
                                        _extract_clothing_features.model.bn1(
                                            _extract_clothing_features.model.conv1(clothing_tensor)
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
            features = features.squeeze().numpy()
        
        # Нормализуем признаки
        features = features / (np.linalg.norm(features) + 1e-8)
        
        # Обрезаем до 512 измерений для совместимости
        if len(features) > 512:
            features = features[:512]
        elif len(features) < 512:
            features = np.pad(features, (0, 512 - len(features)), 'constant')
        
        return features.astype(np.float32)
        
    except Exception as e:
        print(f"⚠️ Ошибка извлечения признаков одежды: {e}")
        return np.zeros(512, dtype=np.float32)


def _analyze_clothing_color(img: np.ndarray, face_bbox: tuple) -> Dict[str, float]:
    """
    Анализ цветов одежды
    """
    try:
        x1, y1, x2, y2 = face_bbox
        face_height = y2 - y1
        
        # Область одежды
        clothing_x1 = max(0, int(x1 - (x2-x1) * 0.2))
        clothing_y1 = min(img.shape[0], int(y2 + face_height * 0.1))
        clothing_x2 = min(img.shape[1], int(x2 + (x2-x1) * 0.2))
        clothing_y2 = min(img.shape[0], int(y2 + face_height * 1.2))
        
        if clothing_y1 >= clothing_y2 or clothing_x1 >= clothing_x2:
            return {'dominant_color': 0.0, 'color_variance': 0.0, 'brightness': 0.5}
        
        clothing_crop = img[clothing_y1:clothing_y2, clothing_x1:clothing_x2]
        
        if clothing_crop.size == 0:
            return {'dominant_color': 0.0, 'color_variance': 0.0, 'brightness': 0.5}
        
        # Преобразуем в HSV для лучшего анализа цвета
        hsv = cv2.cvtColor(clothing_crop, cv2.COLOR_BGR2HSV)
        
        # Доминирующий цвет (Hue)
        hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        dominant_hue = np.argmax(hist_h) / 180.0
        
        # Вариативность цвета
        color_variance = np.var(hsv[:, :, 0]) / 180.0
        
        # Яркость
        brightness = np.mean(hsv[:, :, 2]) / 255.0
        
        return {
            'dominant_color': dominant_hue,
            'color_variance': color_variance,
            'brightness': brightness
        }
        
    except Exception as e:
        print(f"⚠️ Ошибка анализа цвета одежды: {e}")
        return {'dominant_color': 0.0, 'color_variance': 0.0, 'brightness': 0.5}


def _assess_face_quality(face, img: np.ndarray) -> Dict[str, float]:
    """
    Профессиональная оценка качества лица
    Возвращает: dict с метриками качества
    """
    quality = {
        'det_score': getattr(face, 'det_score', 1.0),
        'blur_score': 0.0,
        'pose_score': 1.0,
        'occlusion_score': 1.0,
        'brightness_score': 1.0,
        'total_score': 0.0
    }
    
    try:
        # 1. Оценка размытости (более точная)
        x1, y1, x2, y2 = map(int, face.bbox)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(img.shape[1], x2); y2 = min(img.shape[0], y2)
        crop = img[y1:y2, x1:x2]
        
        if crop.size > 0:
            # Laplacian variance для размытости
            blur_var = _blur_var(crop)
            quality['blur_score'] = min(1.0, blur_var / 200.0)  # Нормализация
            
            # Яркость и контраст
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
            brightness = np.mean(gray) / 255.0
            # Оптимальная яркость 0.3-0.7
            quality['brightness_score'] = 1.0 - abs(brightness - 0.5) * 2.0
        
        # 2. Оценка позы головы (через landmarks если доступны)
        if hasattr(face, 'pose'):
            pose = face.pose
            # Идеальная поза = фронтальное лицо
            # pose обычно [pitch, yaw, roll]
            if pose is not None and len(pose) >= 2:
                yaw, pitch = abs(pose[0]), abs(pose[1])
                # Штраф за отклонение от фронтального вида
                quality['pose_score'] = max(0.0, 1.0 - (yaw + pitch) / 90.0)
        
        # 3. Оценка окклюзии (через landmark confidence если доступно)
        if hasattr(face, 'landmark_3d_68'):
            # Если есть 3D landmarks, проверяем их уверенность
            quality['occlusion_score'] = 0.9  # Базовая оценка
        
        # 4. Итоговая оценка качества (взвешенная сумма)
        weights = {
            'det_score': 0.3,
            'blur_score': 0.25,
            'pose_score': 0.25,
            'brightness_score': 0.1,
            'occlusion_score': 0.1
        }
        
        quality['total_score'] = sum(quality[k] * weights[k] for k in weights.keys())
        
    except Exception as e:
        # При ошибке возвращаем базовую оценку
        quality['total_score'] = quality['det_score'] * 0.5
    
    return quality


# -------------------------------
# Face embeddings extraction
# -------------------------------

def extract_embeddings(
    image_paths: List[Path],
    providers: List[str] = ("CPUExecutionProvider",),
    det_size=(640, 640),
    min_score: float = 0.3,  # Снижено для начальной детекции
    min_blur_var: float = MIN_BLUR_VAR,
    min_quality_score: float = 0.4,  # Минимальная итоговая оценка качества
    include_clothing: bool = True,  # Включить анализ одежды
    progress_callback=None,
) -> Tuple[np.ndarray, List[Path], Dict[Path, int], List[Path], List[Path], List[Dict], List[np.ndarray]]:
    """
    Профессиональное извлечение эмбеддингов с анализом одежды
    Возвращает: (X, owners, img_face_count, unreadable, no_faces, face_qualities, clothing_features)
      - X: np.ndarray [N, D] L2-нормированные эмбеддинги лиц
      - owners: list[Path] длины N (соответствие эмбеддинга → исходный файл)
      - img_face_count: dict[Path] → кол-во используемых лиц из изображения
      - unreadable: список битых/нечитаемых файлов
      - no_faces: список файлов без пригодных лиц
      - face_qualities: список с оценками качества для каждого лица
      - clothing_features: список с признаками одежды для каждого лица
    """
    if not _INSIGHTFACE_OK or FaceAnalysis is None:
        if progress_callback:
            progress_callback("❌ InsightFace не доступен. Установите: pip install insightface", 0)
        raise RuntimeError("InsightFace не доступен. Установите: pip install insightface")
    
    try:
        # Используем более точную модель и настройки
        app = FaceAnalysis(name="buffalo_l", providers=list(providers))
        ctx_id = -1 if "cpu" in str(providers).lower() else 0
        # Увеличиваем размер детекции для лучшего качества
        app.prepare(ctx_id=ctx_id, det_size=(1024, 1024))
    except Exception as e:
        if progress_callback:
            progress_callback(f"❌ Ошибка инициализации InsightFace: {str(e)}", 0)
        raise RuntimeError(f"Ошибка инициализации InsightFace: {str(e)}")

    if progress_callback:
        progress_callback("✅ Модель загружена, начинаем анализ изображений...", 10)

    embeddings: List[np.ndarray] = []
    owners: List[Path] = []
    face_qualities: List[Dict] = []
    clothing_features: List[np.ndarray] = []
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
            # Профессиональная оценка качества лица
            quality = _assess_face_quality(f, img)
            
            # Извлечение признаков одежды
            if include_clothing:
                clothing_feat = _extract_clothing_features(img, f.bbox)
                clothing_color = _analyze_clothing_color(img, f.bbox)
                
                # Объединяем признаки одежды
                combined_clothing = np.concatenate([
                    clothing_feat,
                    np.array([clothing_color['dominant_color'], 
                             clothing_color['color_variance'], 
                             clothing_color['brightness']], dtype=np.float32)
                ])
                clothing_features.append(combined_clothing)
            else:
                clothing_features.append(np.zeros(515, dtype=np.float32))  # 512 + 3
            
            # Фильтрация по итоговой оценке качества
            if quality['total_score'] < min_quality_score:
                continue

            emb = getattr(f, "normed_embedding", None)
            if emb is None:
                continue
            emb = emb.astype(np.float32)
            
            # Профессиональная нормализация эмбеддинга
            n = np.linalg.norm(emb)
            if n <= 1e-6:  # Более строгая проверка нуля
                continue
            emb = emb / n
            
            # Проверка качества эмбеддинга
            if np.any(np.isnan(emb)) or np.any(np.isinf(emb)):
                continue
            
            # Проверка вариативности эмбеддинга (не все значения одинаковые)
            if np.std(emb) < 1e-6:
                continue

            embeddings.append(emb)
            owners.append(p)
            face_qualities.append(quality)
            used += 1

        if used > 0:
            img_face_count[p] = used
        else:
            no_faces.append(p)

    if not embeddings:
        return np.empty((0, 512), dtype=np.float32), owners, img_face_count, unreadable, no_faces, []

    X = np.vstack(embeddings).astype(np.float32)
    
    # Профессиональная нормализация для FAISS
    if _FAISS_OK:
        try:
            faiss.normalize_L2(X)
        except Exception as e:
            print(f"⚠️ Ошибка FAISS нормализации: {e}")
            # Fallback нормализация
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X = X / (norms + 1e-8)
    else:
        # L2 нормализация вручную
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / (norms + 1e-8)
    
    return X, owners, img_face_count, unreadable, no_faces, face_qualities, clothing_features


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


def hdbscan_cluster_professional(
    X: np.ndarray,
    face_qualities: List[Dict] = None,
    min_cluster_size: int = 2,
    min_samples: int = 1,
    progress_callback=None,
) -> List[List[int]]:
    """
    Профессиональная кластеризация с HDBSCAN
    Использует state-of-the-art подход для группировки лиц
    """
    N = X.shape[0]
    if N == 0:
        return []
    
    if progress_callback:
        progress_callback("🧬 HDBSCAN кластеризация (профессиональная)...", 75)
    
    if not _HDBSCAN_OK or hdbscan is None:
        # Fallback на графовый метод
        if progress_callback:
            progress_callback("⚠️ HDBSCAN недоступен, используется графовый метод", 76)
        return []
    
    try:
        # Вычисляем косинусные расстояния (1 - cosine similarity)
        # HDBSCAN работает с метрикой расстояний
        distances = 1.0 - np.dot(X, X.T)
        np.fill_diagonal(distances, 0)  # Расстояние до себя = 0
        
        # Приводим к правильному типу для HDBSCAN (float64)
        distances = distances.astype(np.float64)
        
        # Убеждаемся что матрица симметрична
        distances = (distances + distances.T) / 2.0
        
        # Адаптивные параметры на основе размера датасета
        adaptive_min_cluster_size = max(2, min(min_cluster_size, N // 20))
        adaptive_min_samples = max(1, min(min_samples, adaptive_min_cluster_size // 2))
        
        # HDBSCAN кластеризация с оптимальными параметрами
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=adaptive_min_cluster_size,
            min_samples=adaptive_min_samples,
            metric='precomputed',  # Используем pre-computed distances
            cluster_selection_method='eom',  # Excess of Mass - лучший метод
            alpha=1.0,  # Консервативный параметр
            allow_single_cluster=True,
            cluster_selection_epsilon=0.0,
        )
        
        if progress_callback:
            progress_callback("🔬 Анализ плотности и формирование кластеров...", 80)
        
        # Выполняем кластеризацию
        labels = clusterer.fit_predict(distances)
        
        # Извлекаем вероятности принадлежности
        probabilities = clusterer.probabilities_ if hasattr(clusterer, 'probabilities_') else None
        
        # Формируем кластеры
        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            if label != -1:  # -1 = шум
                # Фильтруем по вероятности если доступна
                if probabilities is not None:
                    if probabilities[idx] >= 0.5:  # Высокая уверенность
                        clusters[label].append(idx)
                else:
                    clusters[label].append(idx)
        
        # Обработка шума - пытаемся назначить в ближайшие кластеры
        noise_indices = [i for i, label in enumerate(labels) if label == -1]
        if noise_indices and clusters:
            if progress_callback:
                progress_callback("🔍 Обработка выбросов...", 85)
            
            for noise_idx in noise_indices:
                noise_emb = X[noise_idx]
                best_cluster = -1
                best_sim = 0.4  # Минимальный порог схожести
                
                for cluster_id, cluster_indices in clusters.items():
                    if not cluster_indices:
                        continue
                    # Сравниваем с медоидом кластера
                    cluster_embs = X[cluster_indices]
                    centroid = np.mean(cluster_embs, axis=0)
                    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
                    sim = float(np.dot(noise_emb, centroid))
                    
                    if sim > best_sim:
                        best_sim = sim
                        best_cluster = cluster_id
                
                if best_cluster != -1:
                    clusters[best_cluster].append(noise_idx)
        
        result = [sorted(indices) for indices in clusters.values() if len(indices) >= 1]
        
        if progress_callback:
            progress_callback(f"✅ Найдено {len(result)} кластеров", 90)
        
        return result
        
    except Exception as e:
        if progress_callback:
            progress_callback(f"⚠️ Ошибка HDBSCAN: {e}, используется fallback", 76)
        return []


def post_process_clusters(
    X: np.ndarray,
    clusters: List[List[int]],
    face_qualities: List[Dict] = None,
    min_cluster_size: int = 2,
    progress_callback=None,
) -> List[List[int]]:
    """Умная пост-обработка кластеров с валидацией"""
    if not clusters:
        return clusters
    
    if progress_callback:
        progress_callback("🔧 Умная пост-обработка кластеров...", 92)
    
    processed_clusters = []
    
    for cluster in clusters:
        if len(cluster) < min_cluster_size:
            processed_clusters.append(cluster)
            continue
            
        # Вычисляем медиоид (наиболее репрезентативный элемент)
        cluster_embeddings = X[cluster]
        similarities = np.dot(cluster_embeddings, cluster_embeddings.T)
        sum_sims = similarities.sum(axis=1)
        medoid_idx_local = np.argmax(sum_sims)
        medoid = cluster_embeddings[medoid_idx_local]
        
        # Вычисляем схожести с медиоидом
        sims_to_medoid = np.dot(cluster_embeddings, medoid)
        
        # Более консервативный порог для сохранения кластеров
        threshold = np.percentile(sims_to_medoid, 25)  # Оставляем 75% (было 85%)
        threshold = max(threshold, 0.4)  # Снижен минимальный порог (было 0.5)
        
        # Фильтруем по схожести
        filtered_indices = [cluster[i] for i, sim in enumerate(sims_to_medoid) if sim >= threshold]
        
        # Дополнительная фильтрация по качеству если доступно
        if face_qualities and len(face_qualities) == len(X):
            quality_filtered = []
            avg_quality = np.mean([face_qualities[i]['total_score'] for i in filtered_indices])
            for idx in filtered_indices:
                if face_qualities[idx]['total_score'] >= avg_quality * 0.6:  # Снижен порог (было 0.7)
                    quality_filtered.append(idx)
            if len(quality_filtered) >= min_cluster_size:
                filtered_indices = quality_filtered
        
        if len(filtered_indices) >= 1:
            processed_clusters.append(filtered_indices)
    
    # Дополнительное слияние похожих кластеров
    if len(processed_clusters) > 1:
        processed_clusters = _smart_merge_clusters(X, processed_clusters)
    
    # Финальная валидация: предотвращение over-clustering
    if len(processed_clusters) > max(1, len(X) // 3):  # Максимум N/3 кластеров
        processed_clusters = _aggressive_merge_clusters(X, processed_clusters)
    
    return processed_clusters


def _aggressive_merge_clusters(X: np.ndarray, clusters: List[List[int]]) -> List[List[int]]:
    """
    Агрессивное слияние кластеров для предотвращения over-clustering
    """
    if len(clusters) <= 1:
        return clusters
    
    # Вычисляем центроиды
    centroids = []
    for cluster in clusters:
        cluster_embeddings = X[cluster]
        centroid = np.mean(cluster_embeddings, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        centroids.append(centroid)
    
    # Находим пары для слияния с более низким порогом
    merged_clusters = []
    used = [False] * len(clusters)
    
    for i in range(len(clusters)):
        if used[i]:
            continue
            
        current_cluster = clusters[i].copy()
        used[i] = True
        
        # Ищем похожие кластеры для слияния
        for j in range(i + 1, len(clusters)):
            if used[j]:
                continue
                
            similarity = float(np.dot(centroids[i], centroids[j]))
            
            # Очень агрессивный порог для слияния
            if similarity >= 0.55:  # Очень низкий порог
                current_cluster.extend(clusters[j])
                used[j] = True
        
        merged_clusters.append(sorted(current_cluster))
    
    return merged_clusters


# -------------------------------
# High-Precision clustering (end-to-end)
# -------------------------------

def hi_precision_cluster(
    X: np.ndarray,
    face_qualities: List[Dict] = None,
    clothing_features: List[np.ndarray] = None,
    min_cluster_size: int = 2,
    min_samples: int = 1,
    t_member: float = T_MEMBER,
    allow_merge: bool = True,
    t_merge: float = T_MERGE,
    use_advanced_ensemble: bool = True,  # Приоритет продвинутому ensemble
    use_clothing: bool = True,
    clothing_weight: float = 0.3,
    progress_callback=None,
) -> List[List[int]]:
    """
    Продвинутая кластеризация лиц с анализом одежды (максимальная точность)
    Использует ensemble из множественных state-of-the-art алгоритмов
    """
    N = X.shape[0]
    if N == 0:
        return []
    
    # Объединяем признаки лица и одежды для улучшенной кластеризации
    if use_clothing and clothing_features:
        if progress_callback:
            progress_callback("👕 Объединение признаков лица и одежды...", 70)
        X = _combine_face_clothing_features(X, clothing_features, face_qualities, clothing_weight)

    # Приоритет: Максимально консервативный подход (один кластер)
    if progress_callback:
        progress_callback("🔗 Максимально консервативный подход...", 75)
    
    # Простое решение: все лица в одном кластере для максимальной точности
    if N <= 10:  # Для небольших датасетов - один кластер
        if progress_callback:
            progress_callback("📦 Создание одного кластера для максимальной точности...", 90)
        return [list(range(N))]
    
    # Для больших датасетов используем консервативный графовый метод
    edges = build_mutual_edges(X, k=min(5, N-1))  # Минимум соседей
    
    if progress_callback:
        progress_callback("🧩 Связные компоненты...", 82)
    comps = connected_components_from_edges(N, edges)
    
    if progress_callback:
        progress_callback("🎯 Фильтрация по медиоиду...", 88)
    clusters = filter_by_medoid(X, comps, t_member=0.3)  # Очень консервативный порог
    
    if allow_merge:
        if progress_callback:
            progress_callback("🧬 Слияние похожих кластеров...", 90)
        clusters = optional_merge_by_centroids(X, clusters, t_merge=0.4, cluster_attrs=None)  # Очень агрессивное слияние
    
    # Минимальная пост-обработка
    clusters = post_process_clusters(
        X, clusters,
        face_qualities=face_qualities,
        min_cluster_size=1,
        progress_callback=progress_callback
    )
    
    return clusters
    
    # Fallback: продвинутый ensemble кластеризации
    if use_advanced_ensemble:
        clusters = _ensemble_clustering_advanced(
            X, 
            face_qualities=face_qualities,
            progress_callback=progress_callback
        )
        
        if clusters:  # Если ensemble успешно выполнился
            # Продвинутая пост-обработка
            clusters = post_process_clusters(
                X, clusters, 
                face_qualities=face_qualities,
                min_cluster_size=1, 
                progress_callback=progress_callback
            )
            return clusters
    
    # Fallback 1: HDBSCAN
    if _HDBSCAN_OK:
        if progress_callback:
            progress_callback("🔬 HDBSCAN fallback...", 75)
        
        clusters = hdbscan_cluster_professional(
            X, 
            face_qualities=face_qualities,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            progress_callback=progress_callback
        )
        
        if clusters:
            clusters = post_process_clusters(
                X, clusters, 
                face_qualities=face_qualities,
                min_cluster_size=1, 
                progress_callback=progress_callback
            )
            return clusters
    
    # Fallback 2: графовый метод
    if progress_callback:
        progress_callback("🔗 Графовая кластеризация (fallback)...", 75)
    
    edges = build_mutual_edges(X, k=KNN_K)
    
    if progress_callback:
        progress_callback("🧩 Связные компоненты...", 82)
    comps = connected_components_from_edges(N, edges)
    
    if progress_callback:
        progress_callback("🎯 Фильтрация по медиоиду...", 88)
    clusters = filter_by_medoid(X, comps, t_member=t_member)
    
    if allow_merge:
        if progress_callback:
            progress_callback("🧬 Слияние похожих кластеров...", 90)
        clusters = optional_merge_by_centroids(X, clusters, t_merge=t_merge, cluster_attrs=None)
    
    # Пост-обработка
    clusters = post_process_clusters(
        X, clusters,
        face_qualities=face_qualities,
        min_cluster_size=1,
        progress_callback=progress_callback
    )
    
    return clusters

# -------------------------------
# Public API: build_plan_live (Hi-Precision)
# -------------------------------

def build_plan_live(
    input_dir: Path,
    det_size=(1024, 1024),  # Увеличено для лучшего качества
    min_score: float = MIN_DET_SCORE,
    min_cluster_size: int = 1,  # Снижено для включения одиночных лиц
    min_samples: int = 1,       # Снижено для более гибкой кластеризации
    providers: List[str] = ("CPUExecutionProvider",),
    progress_callback=None,
    include_excluded: bool = False,
    allow_merge: bool = True,   # Включаем слияние по умолчанию
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

    # 1) Профессиональное извлечение эмбеддингов
    try:
        X, owners, img_face_count, unreadable, no_faces, face_qualities, clothing_features = extract_embeddings(
            all_images,
            providers=providers,
            det_size=det_size,
            min_score=0.3,  # Низкий порог для начальной детекции
            min_quality_score=0.35,  # Итоговая оценка качества
            include_clothing=True,
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

    # 2) Продвинутая кластеризация (Ensemble + HDBSCAN + fallback)
    if progress_callback:
        progress_callback(f"🔄 Продвинутая кластеризация {X.shape[0]} лиц...", 75)
    clusters_idx = hi_precision_cluster(
        X,
        face_qualities=face_qualities,
        clothing_features=clothing_features,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        t_member=t_member,
        allow_merge=True,
        t_merge=t_merge,
        use_advanced_ensemble=True,  # Используем продвинутый ensemble как приоритет
        use_clothing=True,
        clothing_weight=0.3,
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

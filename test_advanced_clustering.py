"""
Тест продвинутой кластеризации
"""
import sys
from pathlib import Path

def test_imports():
    """Тест импортов и зависимостей"""
    print("🧪 Тест 1: Проверка импортов...")
    
    try:
        from cluster_advanced import (
            build_plan_advanced, 
            AdvancedFaceRecognition,
            k_reciprocal_rerank,
            spectral_clustering_with_validation
        )
        print("  ✅ Импорт cluster_advanced успешен")
    except ImportError as e:
        print(f"  ❌ Ошибка импорта cluster_advanced: {e}")
        return False
    
    # Проверка InsightFace
    try:
        from insightface.app import FaceAnalysis
        print("  ✅ InsightFace доступен")
    except ImportError:
        print("  ⚠️  InsightFace не установлен")
        print("     Установите: pip install insightface onnxruntime")
        return False
    
    # Проверка scikit-learn
    try:
        from sklearn.cluster import SpectralClustering
        print("  ✅ scikit-learn доступен")
    except ImportError:
        print("  ❌ scikit-learn не установлен")
        return False
    
    return True

def test_recognizer_init():
    """Тест инициализации распознавателя"""
    print("\n🧪 Тест 2: Инициализация системы распознавания...")
    
    try:
        from cluster_advanced import AdvancedFaceRecognition
        
        recognizer = AdvancedFaceRecognition(
            use_gpu=False,
            min_face_size=20,
            confidence_threshold=0.9
        )
        
        if recognizer.detector_type == 'insightface':
            print("  ✅ InsightFace детектор инициализирован")
            return True
        else:
            print("  ⚠️  InsightFace детектор не инициализирован")
            return False
            
    except Exception as e:
        print(f"  ❌ Ошибка инициализации: {e}")
        return False

def test_quality_assessment():
    """Тест оценки качества"""
    print("\n🧪 Тест 3: Оценка качества изображения...")
    
    try:
        import numpy as np
        from cluster_advanced import calculate_blur_score, calculate_face_quality
        
        # Создаем тестовое изображение
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Тест blur score
        blur = calculate_blur_score(test_img)
        print(f"  ✅ Blur score: {blur:.2f}")
        
        # Тест quality score
        quality = calculate_face_quality(test_img, bbox=(0, 0, 100, 100))
        print(f"  ✅ Quality score: {quality:.3f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Ошибка оценки качества: {e}")
        return False

def test_k_reciprocal():
    """Тест k-reciprocal re-ranking"""
    print("\n🧪 Тест 4: k-reciprocal re-ranking...")
    
    try:
        import numpy as np
        from cluster_advanced import k_reciprocal_rerank
        
        # Создаем тестовую матрицу сходства
        similarity = np.array([
            [1.0, 0.8, 0.3, 0.2],
            [0.8, 1.0, 0.4, 0.1],
            [0.3, 0.4, 1.0, 0.9],
            [0.2, 0.1, 0.9, 1.0]
        ])
        
        reranked = k_reciprocal_rerank(similarity, k=2)
        
        print(f"  ✅ Re-ranking выполнен")
        print(f"     Оригинал [0,1]: {similarity[0,1]:.3f}")
        print(f"     Re-ranked [0,1]: {reranked[0,1]:.3f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Ошибка re-ranking: {e}")
        return False

def test_spectral_clustering():
    """Тест spectral clustering"""
    print("\n🧪 Тест 5: Spectral Clustering...")
    
    try:
        import numpy as np
        from sklearn.preprocessing import normalize
        from cluster_advanced import spectral_clustering_with_validation
        
        # Создаем тестовые эмбеддинги (3 кластера по 5 точек)
        np.random.seed(42)
        cluster1 = np.random.randn(5, 128) + np.array([1, 0] + [0]*126)
        cluster2 = np.random.randn(5, 128) + np.array([0, 1] + [0]*126)
        cluster3 = np.random.randn(5, 128) + np.array([-1, -1] + [0]*126)
        
        embeddings = np.vstack([cluster1, cluster2, cluster3])
        embeddings = normalize(embeddings, norm='l2')
        
        # Кластеризация
        labels = spectral_clustering_with_validation(
            embeddings=[e for e in embeddings],
            n_clusters=3,
            k_reciprocal=2,
            verification_threshold=0.5
        )
        
        unique_labels = len(set(labels) - {-1})
        print(f"  ✅ Clustering завершен: {unique_labels} кластеров")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Ошибка clustering: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Тест интеграции с main.py"""
    print("\n🧪 Тест 6: Интеграция с main.py...")
    
    try:
        import os
        import importlib.util
        
        # Загружаем main.py
        spec = importlib.util.spec_from_file_location("main", "main.py")
        main_module = importlib.util.module_from_spec(spec)
        
        # Проверяем наличие переменных
        spec.loader.exec_module(main_module)
        
        if hasattr(main_module, 'USE_ADVANCED_CLUSTERING'):
            print(f"  ✅ USE_ADVANCED_CLUSTERING: {main_module.USE_ADVANCED_CLUSTERING}")
        else:
            print("  ❌ USE_ADVANCED_CLUSTERING не найден")
            return False
        
        if hasattr(main_module, 'ADVANCED_AVAILABLE'):
            print(f"  ✅ ADVANCED_AVAILABLE: {main_module.ADVANCED_AVAILABLE}")
        else:
            print("  ❌ ADVANCED_AVAILABLE не найден")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Ошибка интеграции: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Запуск всех тестов"""
    print("=" * 60)
    print("🚀 Тестирование продвинутой системы кластеризации")
    print("=" * 60)
    
    tests = [
        ("Импорты", test_imports),
        ("Инициализация", test_recognizer_init),
        ("Оценка качества", test_quality_assessment),
        ("k-reciprocal", test_k_reciprocal),
        ("Spectral Clustering", test_spectral_clustering),
        ("Интеграция", test_integration)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"  ❌ Критическая ошибка: {e}")
            results.append((name, False))
    
    # Итоги
    print("\n" + "=" * 60)
    print("📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print("-" * 60)
    print(f"Пройдено: {passed}/{total} ({passed/total*100:.1f}%)")
    print("=" * 60)
    
    if passed == total:
        print("\n🎉 Все тесты пройдены! Система готова к использованию.")
        print("\n💡 Для запуска с продвинутой кластеризацией:")
        print("   export USE_ADVANCED_CLUSTERING=true")
        print("   python main.py")
        return 0
    else:
        print("\n⚠️  Некоторые тесты не прошли. Проверьте установку зависимостей:")
        print("   pip install -r requirements-advanced.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())


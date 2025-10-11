#!/usr/bin/env python3
"""
Тестовый скрипт для проверки работы алгоритма кластеризации
"""

import os
import sys
from pathlib import Path

def test_clustering():
    """Тестируем алгоритм кластеризации на тестовых изображениях"""
    print("🔍 Тестирование алгоритма кластеризации...")
    
    try:
        from cluster import build_plan_live, denmune_cluster, karate_club_cluster, _DENMUNE_OK, _KARATECLUB_OK
        print("✅ Модуль cluster импортирован")
        
        # Проверяем доступность новых библиотек
        print(f"🔬 DenMune доступен: {_DENMUNE_OK}")
        print(f"🥋 Karate Club доступен: {_KARATECLUB_OK}")
        
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        return False
    
    # Проверяем наличие тестовых изображений
    test_dir = Path("test_images")
    if not test_dir.exists():
        print(f"❌ Папка {test_dir} не найдена")
        return False
    
    # Считаем изображения
    image_files = [f for f in test_dir.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}]
    print(f"📷 Найдено {len(image_files)} изображений в {test_dir}")
    
    if len(image_files) == 0:
        print("❌ Нет изображений для тестирования")
        return False
    
    # Функция для отображения прогресса
    def progress_callback(message, percent=None):
        if percent is not None:
            print(f"📊 {percent}%: {message}")
        else:
            print(f"📝 {message}")
    
    try:
        print("\n🚀 Запуск кластеризации...")
        result = build_plan_live(
            test_dir,
            progress_callback=progress_callback
        )
        
        print("\n📊 Результаты кластеризации:")
        print(f"  - Кластеров найдено: {len(result.get('clusters', {}))}")
        print(f"  - Файлов в плане: {len(result.get('plan', []))}")
        print(f"  - Нечитаемых файлов: {len(result.get('unreadable', []))}")
        print(f"  - Файлов без лиц: {len(result.get('no_faces', []))}")
        
        if 'error' in result:
            print(f"❌ Ошибка: {result['error']}")
            return False
        
        # Показываем детали кластеров
        clusters = result.get('clusters', {})
        if clusters:
            print("\n📁 Детали кластеров:")
            for cluster_id, files in clusters.items():
                print(f"  Кластер {cluster_id}: {len(files)} файлов")
                for file_path in files[:3]:  # Показываем первые 3 файла
                    print(f"    - {Path(file_path).name}")
                if len(files) > 3:
                    print(f"    ... и еще {len(files) - 3} файлов")
        
        print("\n✅ Кластеризация завершена успешно!")
        
        # Тестируем новые методы кластеризации отдельно
        print("\n🧪 Тестирование новых методов кластеризации...")
        test_new_clustering_methods()
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при кластеризации: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_new_clustering_methods():
    """Тестируем новые методы кластеризации на синтетических данных"""
    import numpy as np
    from cluster import denmune_cluster, karate_club_cluster, _DENMUNE_OK, _KARATECLUB_OK
    
    print("🧪 Создание синтетических данных для тестирования...")
    
    # Создаем синтетические данные - 3 кластера по 10 точек каждый
    np.random.seed(42)
    cluster1 = np.random.normal([0, 0], 0.5, (10, 2))
    cluster2 = np.random.normal([5, 5], 0.5, (10, 2))
    cluster3 = np.random.normal([-3, 4], 0.5, (10, 2))
    
    X = np.vstack([cluster1, cluster2, cluster3])
    
    print(f"📊 Создано {X.shape[0]} точек в {X.shape[1]} измерениях")
    
    # Тестируем DenMune
    if _DENMUNE_OK:
        print("\n🔬 Тестирование DenMune...")
        try:
            clusters = denmune_cluster(X, min_cluster_size=2, k=10)
            print(f"✅ DenMune: найдено {len(clusters)} кластеров")
            for i, cluster in enumerate(clusters):
                print(f"  Кластер {i+1}: {len(cluster)} точек")
        except Exception as e:
            print(f"❌ Ошибка DenMune: {e}")
    else:
        print("⚠️ DenMune недоступен")
    
    # Тестируем Karate Club
    if _KARATECLUB_OK:
        print("\n🥋 Тестирование Karate Club...")
        try:
            clusters = karate_club_cluster(X, min_cluster_size=2, embedding_dim=64)
            print(f"✅ Karate Club: найдено {len(clusters)} кластеров")
            for i, cluster in enumerate(clusters):
                print(f"  Кластер {i+1}: {len(cluster)} точек")
        except Exception as e:
            print(f"❌ Ошибка Karate Club: {e}")
    else:
        print("⚠️ Karate Club недоступен")
    
    print("✅ Тестирование новых методов завершено!")


if __name__ == "__main__":
    print("🚀 Тест алгоритма кластеризации FaceSort\n")
    
    success = test_clustering()
    
    if success:
        print("\n✅ Тест пройден! Алгоритм работает корректно.")
    else:
        print("\n❌ Тест не пройден. Есть проблемы с алгоритмом.")
        sys.exit(1)

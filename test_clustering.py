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
        from cluster import build_plan_live
        print("✅ Модуль cluster импортирован")
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
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при кластеризации: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Тест алгоритма кластеризации FaceSort\n")
    
    success = test_clustering()
    
    if success:
        print("\n✅ Тест пройден! Алгоритм работает корректно.")
    else:
        print("\n❌ Тест не пройден. Есть проблемы с алгоритмом.")
        sys.exit(1)

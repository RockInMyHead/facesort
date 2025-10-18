#!/usr/bin/env python3
"""
Диагностический скрипт для проверки кластеризации лиц
Помогает выяснить, почему не создаются папки с людьми
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import face_recognition
from cluster import build_plan_live, distribute_to_folders, IMG_EXTS

def check_images_in_folder(folder_path):
    """Проверяет изображения в папке"""
    folder = Path(folder_path)
    if not folder.exists():
        print(f"❌ Папка не существует: {folder}")
        return False
    
    print(f"📁 Проверяем папку: {folder}")
    
    # Находим все изображения
    images = []
    for ext in IMG_EXTS:
        images.extend(folder.rglob(f"*{ext}"))
        images.extend(folder.rglob(f"*{ext.upper()}"))
    
    print(f"🖼 Найдено изображений: {len(images)}")
    
    if len(images) == 0:
        print("❌ Изображения не найдены!")
        return False
    
    # Проверяем первые несколько изображений
    faces_found = 0
    for i, img_path in enumerate(images[:5]):  # Проверяем первые 5
        print(f"\n🔍 Анализируем: {img_path.name}")
        
        try:
            # Загружаем изображение
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"   ❌ Не удалось загрузить изображение")
                continue
            
            # Конвертируем в RGB для face_recognition
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Ищем лица
            face_locations = face_recognition.face_locations(rgb_image)
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            print(f"   👤 Найдено лиц: {len(face_locations)}")
            print(f"   🧠 Эмбеддингов: {len(face_encodings)}")
            
            if len(face_encodings) > 0:
                faces_found += 1
                print(f"   ✅ Лица обнаружены")
            else:
                print(f"   ❌ Лица не обнаружены")
                
        except Exception as e:
            print(f"   ❌ Ошибка обработки: {e}")
    
    print(f"\n📊 Результат: {faces_found}/{min(5, len(images))} изображений содержат лица")
    return faces_found > 0

def test_clustering(folder_path):
    """Тестирует процесс кластеризации"""
    print(f"\n🔬 Тестируем кластеризацию для: {folder_path}")
    
    try:
        # Строим план кластеризации
        print("📋 Строим план кластеризации...")
        plan = build_plan_live(Path(folder_path))
        
        print(f"📊 Результаты плана:")
        print(f"   - Кластеров: {len(plan.get('clusters', {}))}")
        print(f"   - Файлов в плане: {len(plan.get('plan', []))}")
        print(f"   - Нечитаемых файлов: {len(plan.get('unreadable', []))}")
        print(f"   - Файлов без лиц: {len(plan.get('no_faces', []))}")
        
        if len(plan.get('clusters', {})) == 0:
            print("❌ Кластеры не найдены!")
            print("🔍 Возможные причины:")
            print("   - Недостаточно лиц для кластеризации")
            print("   - Лица слишком разные для группировки")
            print("   - Проблемы с детекцией лиц")
            return False
        
        # Показываем детали кластеров
        clusters = plan.get('clusters', {})
        for cluster_id, files in clusters.items():
            print(f"   📁 Кластер {cluster_id}: {len(files)} файлов")
            for file_path in files[:3]:  # Показываем первые 3 файла
                print(f"      - {Path(file_path).name}")
            if len(files) > 3:
                print(f"      ... и еще {len(files) - 3} файлов")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка кластеризации: {e}")
        return False

def test_distribution(folder_path):
    """Тестирует распределение файлов"""
    print(f"\n📁 Тестируем распределение для: {folder_path}")
    
    try:
        # Строим план
        plan = build_plan_live(Path(folder_path))
        
        if len(plan.get('clusters', {})) == 0:
            print("❌ Нет кластеров для распределения")
            return False
        
        # Тестируем распределение (без реального перемещения)
        print("🔄 Симулируем распределение...")
        
        plan_items = plan.get('plan', [])
        used_clusters = sorted({c for item in plan_items for c in item["cluster"]})
        cluster_id_map = {old: 1 + idx for idx, old in enumerate(used_clusters)}
        
        print(f"📊 Будет создано папок: {len(used_clusters)}")
        for old_id, new_id in cluster_id_map.items():
            cluster_files = [item for item in plan_items if old_id in item["cluster"]]
            print(f"   📁 Папка {new_id}: {len(cluster_files)} файлов")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка распределения: {e}")
        return False

def main():
    if len(sys.argv) != 2:
        print("Использование: python debug_clustering.py <путь_к_папке>")
        print("Пример: python debug_clustering.py /path/to/photos")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    
    print("🔍 Диагностика кластеризации лиц")
    print("=" * 50)
    
    # Шаг 1: Проверяем изображения
    print("1️⃣ Проверка изображений...")
    if not check_images_in_folder(folder_path):
        print("❌ Проблема: нет изображений с лицами")
        sys.exit(1)
    
    # Шаг 2: Тестируем кластеризацию
    print("\n2️⃣ Тестирование кластеризации...")
    if not test_clustering(folder_path):
        print("❌ Проблема: кластеризация не работает")
        sys.exit(1)
    
    # Шаг 3: Тестируем распределение
    print("\n3️⃣ Тестирование распределения...")
    if not test_distribution(folder_path):
        print("❌ Проблема: распределение не работает")
        sys.exit(1)
    
    print("\n✅ Диагностика завершена успешно!")
    print("💡 Если папки все еще не создаются, проверьте:")
    print("   - Права доступа к папке")
    print("   - Логи сервера")
    print("   - Настройки кластеризации")

if __name__ == "__main__":
    main()

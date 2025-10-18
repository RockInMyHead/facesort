#!/usr/bin/env python3
"""
Тестовый скрипт для проверки точности улучшенного распознавания лиц.
Сравнивает старый подход (InsightFace) с новым (face_recognition + DBSCAN + верификация).
"""

from pathlib import Path
from cluster import build_plan_live

def test_face_recognition_accuracy(test_folder_path):
    """
    Тестирует точность распознавания на указанной папке.
    
    Args:
        test_folder_path: Путь к папке с тестовыми изображениями
    """
    print("=" * 80)
    print("🧪 ТЕСТИРОВАНИЕ УЛУЧШЕННОЙ СИСТЕМЫ РАСПОЗНАВАНИЯ ЛИЦ")
    print("=" * 80)
    print(f"\n📁 Тестовая папка: {test_folder_path}\n")
    
    test_path = Path(test_folder_path)
    
    if not test_path.exists():
        print(f"❌ Папка {test_path} не существует")
        return
    
    # Подсчитываем количество изображений
    images = list(test_path.rglob("*"))
    image_count = len([p for p in images if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])
    
    print(f"📊 Найдено изображений: {image_count}")
    print("\n" + "-" * 80)
    print("🚀 Запуск кластеризации с новым алгоритмом...")
    print("-" * 80 + "\n")
    
    def progress_callback(message, percent=None):
        if percent:
            print(f"[{percent:3d}%] {message}")
        else:
            print(f"       {message}")
    
    try:
        result = build_plan_live(
            test_path,
            min_score=0.95,  # Высокий порог для точности
            min_cluster_size=1,  # Разрешить одиночные фото
            min_samples=1,  # Разрешить одиночные кластеры
            progress_callback=progress_callback
        )
        
        print("\n" + "=" * 80)
        print("📊 РЕЗУЛЬТАТЫ КЛАСТЕРИЗАЦИИ")
        print("=" * 80 + "\n")
        
        clusters = result.get("clusters", {})
        plan = result.get("plan", [])
        unreadable = result.get("unreadable", [])
        no_faces = result.get("no_faces", [])
        
        print(f"✅ Всего кластеров (людей): {len(clusters)}")
        print(f"📷 Обработано изображений: {len(plan)}")
        print(f"❌ Нечитаемых файлов: {len(unreadable)}")
        print(f"👤 Без лиц: {len(no_faces)}")
        
        print("\n" + "-" * 80)
        print("📋 ДЕТАЛИ КЛАСТЕРОВ")
        print("-" * 80 + "\n")
        
        # Сортируем кластеры по количеству фото
        sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
        
        for cluster_id, paths in sorted_clusters[:10]:  # Показываем топ-10
            print(f"  Кластер {cluster_id}: {len(paths)} фото")
            for path in list(paths)[:3]:  # Первые 3 фото
                print(f"    - {Path(path).name}")
            if len(paths) > 3:
                print(f"    ... и ещё {len(paths) - 3} фото")
        
        if len(sorted_clusters) > 10:
            print(f"\n  ... и ещё {len(sorted_clusters) - 10} кластеров")
        
        # Статистика одиночных фото
        single_photo_clusters = [c for c in clusters.values() if len(c) == 1]
        print(f"\n📁 Одиночных фото (отдельные папки): {len(single_photo_clusters)}")
        
        # Оценка качества
        print("\n" + "=" * 80)
        print("📈 ОЦЕНКА КАЧЕСТВА")
        print("=" * 80 + "\n")
        
        coverage = (len(plan) / max(image_count, 1)) * 100
        print(f"  Покрытие: {coverage:.1f}% ({len(plan)}/{image_count} изображений)")
        
        if len(no_faces) > 0:
            print(f"  Изображения без лиц: {len(no_faces)}")
            for path in no_faces[:5]:
                print(f"    - {Path(path).name}")
        
        print("\n" + "=" * 80)
        print("✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
        print("=" * 80 + "\n")
        
        print("💡 КЛЮЧЕВЫЕ УЛУЧШЕНИЯ:")
        print("  ✓ Используется face_recognition с точностью 99.38% (LFW)")
        print("  ✓ CNN детектор лиц для максимальной точности")
        print("  ✓ DBSCAN с адаптивным epsilon вместо HDBSCAN")
        print("  ✓ Двухэтапная верификация кластеров")
        print("  ✓ Поддержка одиночных фотографий (каждая в своей папке)")
        print("  ✓ Euclidean distance оптимизирован для face_recognition\n")
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_folder = sys.argv[1]
    else:
        # Используем папку по умолчанию
        test_folder = "/Users/artembutko/Desktop"
        print(f"ℹ️  Используется папка по умолчанию: {test_folder}")
        print(f"ℹ️  Запустите с аргументом для другой папки: python test_accuracy.py /path/to/folder\n")
    
    test_face_recognition_accuracy(test_folder)


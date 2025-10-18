#!/usr/bin/env python3
"""
Простой тест кластеризации для диагностики проблемы
"""

import sys
from pathlib import Path
from cluster_improved import build_plan_live, distribute_to_folders

def test_simple_clustering(folder_path):
    """Простой тест кластеризации"""
    print(f"🔍 Тестируем кластеризацию для: {folder_path}")
    
    try:
        # Строим план
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
        
        # Тестируем распределение (без реального перемещения)
        print("\n🔄 Тестируем распределение...")
        
        plan_items = plan.get('plan', [])
        used_clusters = sorted({c for item in plan_items for c in item["cluster"]})
        cluster_id_map = {old: 1 + idx for idx, old in enumerate(used_clusters)}
        
        print(f"📊 Будет создано папок: {len(used_clusters)}")
        for old_id, new_id in cluster_id_map.items():
            cluster_files = [item for item in plan_items if old_id in item["cluster"]]
            print(f"   📁 Папка {new_id}: {len(cluster_files)} файлов")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка кластеризации: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    if len(sys.argv) != 2:
        print("Использование: python test_clustering_simple.py <путь_к_папке>")
        print("Пример: python test_clustering_simple.py /path/to/photos")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    
    print("🔍 Простой тест кластеризации")
    print("=" * 50)
    
    if test_clustering_simple(folder_path):
        print("\n✅ Тест завершен успешно!")
        print("💡 Если папки все еще не создаются, проверьте:")
        print("   - Права доступа к папке")
        print("   - Логи сервера")
        print("   - Настройки кластеризации")
    else:
        print("\n❌ Тест не прошел!")
        sys.exit(1)

if __name__ == "__main__":
    main()

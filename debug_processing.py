#!/usr/bin/env python3
"""
Диагностика проблемы с обработкой
"""

import requests
import json
import time
import os
from pathlib import Path

def debug_processing():
    """Диагностирует проблему с обработкой"""
    base_url = "http://localhost:8000"
    
    print("🔍 Диагностика проблемы с обработкой...")
    
    # Создаем тестовую папку
    test_folder = Path("test_photos")
    test_folder.mkdir(exist_ok=True)
    
    # Создаем тестовые файлы (пустые, но с правильными расширениями)
    test_files = ["test1.jpg", "test2.jpg", "test3.jpg"]
    for filename in test_files:
        test_file = test_folder / filename
        if not test_file.exists():
            test_file.write_text("test image content")
    
    print(f"📁 Создана тестовая папка: {test_folder.absolute()}")
    print(f"📄 Тестовые файлы: {test_files}")
    
    # Проверяем содержимое папки
    try:
        response = requests.get(f"{base_url}/api/folder", params={"path": str(test_folder.absolute())})
        if response.status_code == 200:
            folder_data = response.json()
            print(f"✅ Папка доступна: {len(folder_data['folders'])} папок, {len(folder_data['files'])} файлов")
        else:
            print(f"❌ Папка недоступна: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Ошибка проверки папки: {e}")
        return
    
    # Добавляем папку в очередь
    try:
        queue_data = {"path": str(test_folder.absolute())}
        response = requests.post(f"{base_url}/api/queue/add", json=queue_data, params={"includeExcluded": True})
        
        if response.status_code == 200:
            print("✅ Папка добавлена в очередь")
        else:
            print(f"❌ Ошибка добавления в очередь: {response.status_code} - {response.text}")
            return
    except Exception as e:
        print(f"❌ Ошибка добавления в очередь: {e}")
        return
    
    # Проверяем очередь
    try:
        response = requests.get(f"{base_url}/api/queue")
        queue_data = response.json()
        print(f"📋 Очередь: {queue_data['queue']}")
    except Exception as e:
        print(f"❌ Ошибка проверки очереди: {e}")
    
    # Запускаем обработку
    try:
        print("🚀 Запускаем обработку...")
        response = requests.post(f"{base_url}/api/queue/process", params={"includeExcluded": True})
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Обработка запущена: {result['message']}")
            print(f"📋 ID задач: {result['task_ids']}")
            
            # Отслеживаем прогресс
            task_ids = result['task_ids']
            for task_id in task_ids:
                print(f"\n🔍 Отслеживаем задачу {task_id}...")
                
                for i in range(10):  # Ждем до 10 секунд
                    try:
                        response = requests.get(f"{base_url}/api/tasks/{task_id}")
                        if response.status_code == 200:
                            task = response.json()
                            status = task['status']
                            progress = task['progress']
                            message = task['message']
                            
                            print(f"   📊 {status}: {progress}% - {message}")
                            
                            if status == "completed":
                                print(f"✅ Задача завершена!")
                                if 'result' in task:
                                    result_data = task['result']
                                    print(f"   📁 Создано кластеров: {result_data.get('clusters_count', 0)}")
                                    print(f"   📤 Перемещено файлов: {result_data.get('moved', 0)}")
                                    print(f"   📋 Скопировано файлов: {result_data.get('copied', 0)}")
                                break
                            elif status == "error":
                                print(f"❌ Ошибка в задаче: {message}")
                                break
                        else:
                            print(f"❌ Ошибка получения статуса: {response.status_code}")
                            break
                    except Exception as e:
                        print(f"❌ Ошибка отслеживания: {e}")
                        break
                    
                    time.sleep(1)
                else:
                    print("⏰ Таймаут ожидания задачи")
        else:
            print(f"❌ Ошибка запуска обработки: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Ошибка запуска обработки: {e}")

if __name__ == "__main__":
    debug_processing()

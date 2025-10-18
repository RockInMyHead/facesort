#!/usr/bin/env python3
"""
Тест API для проверки работы кластеризации
"""

import requests
import json
import time

def test_api():
    """Тестирует API кластеризации"""
    base_url = "http://localhost:8000"
    
    print("🔍 Тестируем API кластеризации...")
    
    # Тест 1: Проверяем доступность сервера
    try:
        response = requests.get(f"{base_url}/api/tasks")
        print(f"✅ Сервер доступен: {response.status_code}")
        print(f"📊 Текущие задачи: {response.json()}")
    except Exception as e:
        print(f"❌ Сервер недоступен: {e}")
        return
    
    # Тест 2: Проверяем диски
    try:
        response = requests.get(f"{base_url}/api/drives")
        drives = response.json()
        print(f"✅ Найдено дисков: {len(drives['folders'])}")
        for drive in drives['folders'][:3]:
            print(f"   💽 {drive['name']}: {drive['image_count']} изображений")
    except Exception as e:
        print(f"❌ Ошибка получения дисков: {e}")
    
    # Тест 3: Проверяем папку (если есть)
    try:
        # Попробуем получить содержимое домашней папки
        home_path = "/Users/artembutko"
        response = requests.get(f"{base_url}/api/folder", params={"path": home_path})
        if response.status_code == 200:
            folder_data = response.json()
            print(f"✅ Папка {home_path}: {len(folder_data['folders'])} папок, {len(folder_data['files'])} файлов")
        else:
            print(f"⚠️ Папка {home_path} недоступна: {response.status_code}")
    except Exception as e:
        print(f"❌ Ошибка проверки папки: {e}")
    
    print("\n🎯 Для тестирования кластеризации:")
    print("1. Откройте http://localhost:8000 в браузере")
    print("2. Выберите папку с фотографиями")
    print("3. Добавьте в очередь")
    print("4. Запустите обработку")
    print("5. Следите за логами в консоли")

if __name__ == "__main__":
    test_api()

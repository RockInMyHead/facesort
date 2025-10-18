#!/usr/bin/env python3
"""
Диагностика проблемы с распознаванием лиц
"""

import cv2
import numpy as np
import face_recognition
from pathlib import Path

def test_face_recognition():
    """Тестирует распознавание лиц на изображениях"""
    print("🔍 Тестируем распознавание лиц...")
    
    test_folder = Path("test_photos")
    
    if not test_folder.exists():
        print("❌ Папка test_photos не существует")
        return
    
    # Находим все изображения
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_files.extend(test_folder.glob(f"*{ext}"))
        image_files.extend(test_folder.glob(f"*{ext.upper()}"))
    
    print(f"📁 Найдено изображений: {len(image_files)}")
    
    if len(image_files) == 0:
        print("❌ Изображения не найдены")
        return
    
    total_faces = 0
    processed_images = 0
    
    for img_path in image_files:
        print(f"\n🔍 Анализируем: {img_path.name}")
        
        try:
            # Загружаем изображение
            image = face_recognition.load_image_file(str(img_path))
            print(f"   📏 Размер изображения: {image.shape}")
            
            # Ищем лица
            face_locations = face_recognition.face_locations(image, model="cnn")
            print(f"   👤 Найдено лиц: {len(face_locations)}")
            
            if len(face_locations) > 0:
                # Извлекаем эмбеддинги
                face_encodings = face_recognition.face_encodings(image, face_locations, model="large")
                print(f"   🧠 Извлечено эмбеддингов: {len(face_encodings)}")
                
                total_faces += len(face_encodings)
                processed_images += 1
                
                # Показываем координаты лиц
                for i, (top, right, bottom, left) in enumerate(face_locations):
                    print(f"      Лицо {i+1}: ({left}, {top}) - ({right}, {bottom})")
            else:
                print(f"   ❌ Лица не найдены")
                
        except Exception as e:
            print(f"   ❌ Ошибка обработки: {e}")
    
    print(f"\n📊 Результат:")
    print(f"   📁 Обработано изображений: {processed_images}/{len(image_files)}")
    print(f"   👤 Всего найдено лиц: {total_faces}")
    
    if total_faces == 0:
        print("\n❌ Проблема: Лица не найдены!")
        print("💡 Возможные причины:")
        print("   - Изображения не содержат лиц")
        print("   - Лица слишком маленькие")
        print("   - Плохое качество изображения")
        print("   - Проблемы с библиотекой face_recognition")
    elif total_faces < 2:
        print("\n⚠️ Проблема: Недостаточно лиц для кластеризации!")
        print("💡 Нужно минимум 2 лица для создания кластеров")
    else:
        print(f"\n✅ Найдено достаточно лиц для кластеризации: {total_faces}")

def test_opencv():
    """Тестирует OpenCV для детекции лиц"""
    print("\n🔍 Тестируем OpenCV...")
    
    # Загружаем каскад Хаара
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    test_folder = Path("test_photos")
    image_files = list(test_folder.glob("*.png")) + list(test_folder.glob("*.jpg"))
    
    for img_path in image_files[:2]:  # Тестируем первые 2
        print(f"\n🔍 OpenCV анализ: {img_path.name}")
        
        try:
            # Загружаем изображение
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"   ❌ Не удалось загрузить изображение")
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            print(f"   👤 OpenCV нашел лиц: {len(faces)}")
            
            for i, (x, y, w, h) in enumerate(faces):
                print(f"      Лицо {i+1}: ({x}, {y}) размер {w}x{h}")
                
        except Exception as e:
            print(f"   ❌ Ошибка OpenCV: {e}")

if __name__ == "__main__":
    test_face_recognition()
    test_opencv()

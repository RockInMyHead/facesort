#!/usr/bin/env python3
"""
Простой тест распознавания лиц
"""

import face_recognition
from pathlib import Path

def test_simple():
    """Простой тест"""
    print("🔍 Простой тест распознавания лиц...")
    
    test_folder = Path("test_photos")
    image_files = list(test_folder.glob("*.png"))
    
    print(f"📁 Найдено PNG файлов: {len(image_files)}")
    
    for img_path in image_files:
        print(f"\n🔍 Тестируем: {img_path.name}")
        
        try:
            # Загружаем изображение
            image = face_recognition.load_image_file(str(img_path))
            print(f"   ✅ Изображение загружено: {image.shape}")
            
            # Ищем лица
            face_locations = face_recognition.face_locations(image)
            print(f"   👤 Найдено лиц: {len(face_locations)}")
            
            if len(face_locations) > 0:
                face_encodings = face_recognition.face_encodings(image, face_locations)
                print(f"   🧠 Эмбеддингов: {len(face_encodings)}")
            else:
                print(f"   ❌ Лица не найдены")
                
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")

if __name__ == "__main__":
    test_simple()

#!/usr/bin/env python3
"""
Тестовый скрипт для проверки зависимостей
"""

def test_imports():
    """Тестируем импорты всех зависимостей"""
    print("🔍 Проверка зависимостей...")
    
    # Проверяем основные библиотеки
    try:
        import numpy as np
        print("✅ NumPy:", np.__version__)
    except ImportError as e:
        print("❌ NumPy не установлен:", e)
        return False
    
    try:
        import cv2
        print("✅ OpenCV:", cv2.__version__)
    except ImportError as e:
        print("❌ OpenCV не установлен:", e)
        return False
    
    try:
        import sklearn
        print("✅ Scikit-learn:", sklearn.__version__)
    except ImportError as e:
        print("❌ Scikit-learn не установлен:", e)
        return False
    
    # Проверяем InsightFace
    try:
        from insightface.app import FaceAnalysis
        print("✅ InsightFace доступен")
        
        # Пытаемся инициализировать
        try:
            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=(640, 640))
            print("✅ InsightFace модель загружена успешно")
        except Exception as e:
            print("⚠️ InsightFace модель не загружается:", e)
            return False
            
    except ImportError as e:
        print("❌ InsightFace не установлен:", e)
        print("   Установите: pip install insightface")
        return False
    
    # Проверяем опциональные библиотеки
    try:
        import faiss
        print("✅ FAISS доступен")
    except ImportError:
        print("⚠️ FAISS не установлен (опционально)")
    
    try:
        import networkx as nx
        print("✅ NetworkX доступен")
    except ImportError:
        print("⚠️ NetworkX не установлен (опционально)")
    
    print("\n🎉 Все основные зависимости работают!")
    return True

def test_cluster_module():
    """Тестируем модуль cluster"""
    print("\n🔍 Тестирование модуля cluster...")
    
    try:
        from cluster import build_plan_live, _INSIGHTFACE_OK
        print("✅ Модуль cluster импортирован")
        print(f"✅ InsightFace статус: {_INSIGHTFACE_OK}")
        
        if _INSIGHTFACE_OK:
            print("✅ Готов к кластеризации лиц")
        else:
            print("❌ InsightFace недоступен")
            return False
            
    except ImportError as e:
        print("❌ Ошибка импорта cluster:", e)
        return False
    except Exception as e:
        print("❌ Ошибка в cluster:", e)
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 Тест зависимостей FaceSort\n")
    
    success = test_imports()
    if success:
        success = test_cluster_module()
    
    if success:
        print("\n✅ Все тесты пройдены! Приложение готово к работе.")
    else:
        print("\n❌ Есть проблемы с зависимостями. Проверьте установку.")
        print("\nДля установки всех зависимостей выполните:")
        print("pip install -r requirements.txt")

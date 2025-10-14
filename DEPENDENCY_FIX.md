# 🔧 Решение конфликтов зависимостей

## Проблема
При установке зависимостей могут возникнуть конфликты версий, особенно с `protobuf` и `mediapipe`.

## ✅ Решение

### 1. Используйте виртуальное окружение
```bash
# Создайте виртуальное окружение
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# или
.venv\Scripts\activate  # Windows
```

### 2. Установите зависимости в правильном порядке
```bash
# Сначала основные зависимости
pip install fastapi uvicorn python-multipart pydantic pillow psutil

# Затем машинное обучение
pip install numpy==1.26.4 opencv-python scikit-learn hdbscan

# Затем распознавание лиц
pip install face-recognition dlib

# И наконец MediaPipe
pip install mediapipe
```

### 3. Если возникают конфликты с protobuf
```bash
# Обновите protobuf до совместимой версии
pip install --upgrade protobuf==4.25.8

# Или используйте фиксированные версии
pip install -r requirements-fixed.txt
```

### 4. Альтернативное решение - используйте requirements-fixed.txt
```bash
# Удалите текущие пакеты
pip uninstall -y protobuf mediapipe opencv-contrib-python

# Установите фиксированные версии
pip install -r requirements-fixed.txt
```

## 🚨 Частые проблемы и решения

### Проблема: "protobuf version conflict"
**Решение:**
```bash
pip install --upgrade protobuf==4.25.8
```

### Проблема: "onnx requires protobuf>=4.25.1"
**Решение:**
```bash
pip install protobuf>=4.25.1
```

### Проблема: "MediaPipe не импортируется"
**Решение:**
```bash
# Убедитесь что используете виртуальное окружение
source .venv/bin/activate
pip install mediapipe==0.10.21
```

## 📋 Проверка установки
```bash
# Проверьте что все работает
python -c "import face_recognition; print('✅ face_recognition работает!')"
python -c "import mediapipe; print('✅ mediapipe работает!')"
python -c "import fastapi; print('✅ fastapi работает!')"
```

## 🔄 Полная переустановка
Если ничего не помогает:
```bash
# Удалите виртуальное окружение
rm -rf .venv

# Создайте новое
python3 -m venv .venv
source .venv/bin/activate

# Установите фиксированные версии
pip install -r requirements-fixed.txt
```

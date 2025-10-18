@echo off
echo 🚀 Запуск установки зависимостей для Windows (без dlib)...

REM Проверяем наличие Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Python не найден. Пожалуйста, установите Python 3.9+ и добавьте его в PATH.
    echo https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Создаем виртуальное окружение
echo 🛠️ Создание виртуального окружения...
python -m venv venv
if %errorlevel% neq 0 (
    echo ❌ Ошибка при создании виртуального окружения.
    pause
    exit /b 1
)

REM Активируем виртуальное окружение
echo 🔛 Активация виртуального окружения...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ❌ Ошибка при активации виртуального окружения.
    pause
    exit /b 1
)

REM Обновляем pip
echo 🔄 Обновление pip...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo ❌ Ошибка при обновлении pip.
    pause
    exit /b 1
)

REM Устанавливаем зависимости
echo 📦 Установка зависимостей из requirements_windows_simple.txt...
pip install -r requirements_windows_simple.txt
if %errorlevel% neq 0 (
    echo ❌ Ошибка при установке зависимостей.
    pause
    exit /b 1
)

echo.
echo ✅ Установка завершена!
echo.
echo Для запуска сервера:
echo 1. Убедитесь, что виртуальное окружение активировано: call venv\Scripts\activate.bat
echo 2. Запустите: python main_windows_simple.py
echo.
pause
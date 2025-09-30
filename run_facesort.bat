@echo off
setlocal

REM Перейти в папку скрипта (учитывает пробелы и скобки в пути)
pushd "%~dp0"

REM Обеспечить кодировку UTF-8 в консоли Python
set PYTHONIOENCODING=utf-8

REM Проверка наличия Python launcher
where py >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python launcher 'py' не найден. Установите Python с опцией "Add to PATH" и попробуйте снова.
  pause
  exit /b 1
)

REM Создать venv, если отсутствует
if not exist "venv\Scripts\python.exe" (
  echo [INFO] Создаю виртуальное окружение...
  py -m venv venv
  if errorlevel 1 (
    echo [ERROR] Не удалось создать виртуальное окружение.
    pause
    exit /b 1
  )
)

REM Активировать venv
call "venv\Scripts\activate.bat"
if errorlevel 1 (
  echo [ERROR] Не удалось активировать виртуальное окружение.
  pause
  exit /b 1
)

REM Обновить pip
python -m pip install --upgrade pip

REM Установить зависимости
if exist "requirements.txt" (
  echo [INFO] Устанавливаю зависимости из requirements.txt ...
  python -m pip install -r requirements.txt
  if errorlevel 1 (
    echo [ERROR] Установка зависимостей завершилась с ошибкой.
    pause
    exit /b 1
  )
) else (
  echo [WARN] Файл requirements.txt не найден. Пропускаю установку зависимостей.
)

REM Проверка наличия uvicorn
python -c "import uvicorn" 2>nul
if errorlevel 1 (
  echo [INFO] Устанавливаю uvicorn...
  python -m pip install uvicorn
)

REM Запуск сервера (доступно на http://localhost:8000)
echo [INFO] Запускаю сервер на http://localhost:8000 ...
start "" "http://localhost:8000"

python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

REM Держать окно открытым, если сервер завершился с ошибкой
if errorlevel 1 (
  echo.
  echo [ERROR] Сервер завершился. Нажмите любую клавишу для выхода...
  pause >nul
)

REM Держать окно открытым и после успешного завершения (на всякий случай)
echo.
echo [INFO] Окно останется открытым. Нажмите любую клавишу для выхода...
pause >nul

popd
endlocal



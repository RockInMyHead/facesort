@echo off
setlocal

REM Запуск установщика в PowerShell с обходом ExecutionPolicy и удержанием окна
pushd "%~dp0"

echo [INFO] Запускаю PowerShell установщик (install_windows.ps1)...
powershell -NoProfile -ExecutionPolicy Bypass -File ".\install_windows.ps1"

echo.
echo [INFO] Готово. Нажмите любую клавишу для выхода...
pause >nul

popd
endlocal



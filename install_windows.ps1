#Requires -Version 5.1
param(
    [switch]$NoWinget
)

$ErrorActionPreference = 'Continue'

Write-Host "=== Facesort Windows Setup ===" -ForegroundColor Cyan
Write-Host "Текущая директория: $PWD" -ForegroundColor DarkGray

function Test-Command($name) {
    try { Get-Command $name -ErrorAction Stop | Out-Null; return $true } catch { return $false }
}

function Ensure-Winget() {
    if ($NoWinget) { return $false }
    if (Test-Command winget) { return $true }
    Write-Warning "winget не найден. Установите Microsoft App Installer из Microsoft Store или запустите с -NoWinget"
    return $false
}

function Winget-Install($id, $name) {
    if (-not (Ensure-Winget)) { return }
    Write-Host "[winget] Установка: $name ..." -ForegroundColor Yellow
    try { winget install --id $id --silent --accept-source-agreements --accept-package-agreements | Out-Null } catch { Write-Warning "winget: пропуск $name ($_ )" }
}

# 1) Базовые инструменты
if (Ensure-Winget) {
    Winget-Install 'Python.Python.3.11' 'Python 3.11'
    Winget-Install 'Git.Git' 'Git'
    Winget-Install 'Microsoft.VCRedist.2015+.x64' 'Microsoft VC++ Redistributable x64'
}

# 2) Проверка Python
$py = $null
if (Test-Command py) { $py = 'py' }
elseif (Test-Command python) { $py = 'python' }
if (-not $py) { throw "Python не найден. Установите Python и добавьте в PATH." }

# 3) Создание и активация venv
Write-Host "[python] Настраиваю виртуальное окружение..." -ForegroundColor Yellow
try { & $py -m venv venv } catch { Write-Warning "Не удалось создать venv: $_" }

$venvActivate = Join-Path $PWD 'venv\Scripts\Activate.ps1'
if (-not (Test-Path $venvActivate)) { throw "Не найдено venv Activate.ps1" }
try { . $venvActivate } catch { Write-Warning "Не удалось активировать venv: $_" }

# 4) Обновление pip и установка зависимостей
Write-Host "[pip] Обновление pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip wheel setuptools || Write-Warning "pip upgrade failed"

if (Test-Path 'requirements.txt') {
    Write-Host "[pip] Установка зависимостей из requirements.txt..." -ForegroundColor Yellow
    python -m pip install -r requirements.txt || Write-Warning "requirements install failed"
}

# 5) Установка ключевых пакетов (на случай отсутствия в requirements)
Write-Host "[pip] Проверка и установка ключевых пакетов..." -ForegroundColor Yellow
python - << 'PY'
import sys, subprocess
def ensure(pkg):
    try:
        __import__(pkg.split('==')[0].split('[')[0])
    except Exception:
        subprocess.call([sys.executable, '-m', 'pip', 'install', pkg])

for p in [
    'fastapi',
    'uvicorn',
    'psutil',
    'insightface',
    'onnxruntime',
    'opencv-python',
    'hdbscan',
    'pillow'
]:
    ensure(p)
print('OK')
PY

# 6) Предзагрузка моделей InsightFace (ускоряет первый запуск)
Write-Host "[insightface] Предзагрузка моделей..." -ForegroundColor Yellow
python - << 'PY'
from insightface.app import FaceAnalysis
try:
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=-1, det_size=(640,640))
except Exception as e:
    print('Skip model preload:', e)
print('Models ready')
PY

Write-Host "=== Готово. Запуск: run_facesort.bat ===" -ForegroundColor Green
try {
    # Оставить окно открытым, чтобы пользователь видел результат
    Read-Host "Нажмите Enter для выхода"
} catch {}


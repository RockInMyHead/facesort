#Requires -Version 5.1
param(
    [switch]$NoWinget
)

$ErrorActionPreference = 'Stop'

Write-Host "=== Facesort Windows Setup ===" -ForegroundColor Cyan

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
    winget install --id $id --silent --accept-source-agreements --accept-package-agreements | Out-Null
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
& $py -m venv venv

$venvActivate = Join-Path $PWD 'venv\Scripts\Activate.ps1'
if (-not (Test-Path $venvActivate)) { throw "Не найдено venv Activate.ps1" }
. $venvActivate

# 4) Обновление pip и установка зависимостей
Write-Host "[pip] Обновление pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip wheel setuptools

if (Test-Path 'requirements.txt') {
    Write-Host "[pip] Установка зависимостей из requirements.txt..." -ForegroundColor Yellow
    python -m pip install -r requirements.txt
}

# 5) Установка ключевых пакетов (на случай отсутствия в requirements)
Write-Host "[pip] Проверка и установка ключевых пакетов..." -ForegroundColor Yellow
python - << 'PY'
import sys, subprocess
def ensure(pkg):
    try:
        __import__(pkg.split('==')[0].split('[')[0])
    except Exception:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])

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
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=-1, det_size=(640,640))
print('Models ready')
PY

Write-Host "=== Готово. Запуск: run_facesort.bat ===" -ForegroundColor Green
try {
    # Оставить окно открытым, чтобы пользователь видел результат
    Read-Host "Нажмите Enter для выхода"
} catch {}


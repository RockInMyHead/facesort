#!/usr/bin/env python3
"""
Скрипт диагностики для Windows
Проверяет доступность сервера и сетевые настройки
"""

import socket
import subprocess
import sys
import requests
from pathlib import Path

def check_port_open(host, port):
    """Проверяет, открыт ли порт"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False

def get_local_ip():
    """Получает локальный IP адрес"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

def check_server_response(url):
    """Проверяет ответ сервера"""
    try:
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    print("🔍 Диагностика FaceSort на Windows")
    print("=" * 50)
    
    # Проверка порта 8000
    print("1. Проверка порта 8000...")
    if check_port_open("127.0.0.1", 8000):
        print("   ✅ Порт 8000 открыт локально")
    else:
        print("   ❌ Порт 8000 не открыт локально")
    
    if check_port_open("0.0.0.0", 8000):
        print("   ✅ Порт 8000 открыт для всех интерфейсов")
    else:
        print("   ❌ Порт 8000 не открыт для всех интерфейсов")
    
    # Проверка HTTP ответа
    print("\n2. Проверка HTTP ответа...")
    urls = [
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        f"http://{get_local_ip()}:8000"
    ]
    
    for url in urls:
        if check_server_response(url):
            print(f"   ✅ Сервер отвечает на {url}")
        else:
            print(f"   ❌ Сервер не отвечает на {url}")
    
    # Информация о сети
    print("\n3. Сетевая информация...")
    local_ip = get_local_ip()
    print(f"   📍 Локальный IP: {local_ip}")
    print(f"   🌐 Попробуйте: http://{local_ip}:8000")
    
    # Проверка процессов
    print("\n4. Проверка процессов...")
    try:
        result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True)
        if ':8000' in result.stdout:
            print("   ✅ Процесс слушает порт 8000")
            lines = [line for line in result.stdout.split('\n') if ':8000' in line]
            for line in lines:
                print(f"   📋 {line.strip()}")
        else:
            print("   ❌ Нет процессов на порту 8000")
    except:
        print("   ⚠️ Не удалось проверить процессы")
    
    # Рекомендации
    print("\n5. Рекомендации:")
    print("   🔧 Если порт не открыт:")
    print("      - Проверьте, что сервер запущен")
    print("      - Проверьте брандмауэр Windows")
    print("      - Запустите как администратор")
    print("   🔧 Если сервер не отвечает:")
    print("      - Проверьте логи сервера")
    print("      - Попробуйте другой порт")
    print("      - Проверьте антивирус")
    
    print("\n📋 Попробуйте эти адреса:")
    print(f"   http://127.0.0.1:8000")
    print(f"   http://localhost:8000")
    print(f"   http://{local_ip}:8000")

if __name__ == "__main__":
    main()

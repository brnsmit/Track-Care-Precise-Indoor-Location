# ============================================================================
# 15. SCRIPT DE TESTE
# ============================================================================

test_code = '''
# ============================================================================
# TESTE DA API TRACK&CARE
# ============================================================================

import requests
import json

BASE_URL = "http://localhost:8000"

# Teste 1: Health Check
print("=" * 60)
print("TESTE 1: Health Check")
print("=" * 60)
response = requests.get(f"{BASE_URL}/health")
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}\\n")

# Teste 2: Predição Única
print("=" * 60)
print("TESTE 2: Predição Única")
print("=" * 60)

test_data = {
    "imei": "IMEI_IPHONE_X",
    "sensor_latlong": "-22.809788, -47.059685",
    "sensor_room": "Enfermaria",
    "rssi": -71.76
}

response = requests.post(f"{BASE_URL}/predict", json=test_data)
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}\\n")

# Teste 3: RSSI Inválido
print("=" * 60)
print("TESTE 3: RSSI Inválido (Deve Falhar)")
print("=" * 60)

test_data_invalid = {
    "imei": "IMEI_IPHONE_X",
    "sensor_latlong": "-22.809788, -47.059685",
    "sensor_room": "Enfermaria",
    "rssi": 50  # Inválido!
}

response = requests.post(f"{BASE_URL}/predict", json=test_data_invalid)
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}\\n")

# Teste 4: Batch
print("=" * 60)
print("TESTE 4: Predição em Batch")
print("=" * 60)

batch_data = [
    {
        "imei": "IMEI_IPHONE_X",
        "sensor_latlong": "-22.809788, -47.059685",
        "sensor_room": "Enfermaria",
        "rssi": -71.76
    },
    {
        "imei": "IMEI_SAMSUNG_S23",
        "sensor_latlong": "-22.809713, -47.059590",
        "sensor_room": "Enfermaria",
        "rssi": -84.13
    },
    {
        "imei": "IMEI_MOTO_G_OLD",
        "sensor_latlong": "-22.809683, -47.059694",
        "sensor_room": "Corredor",
        "rssi": -82.86
    }
]

response = requests.post(f"{BASE_URL}/predict/batch", json=batch_data)
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}\\n")

print("=" * 60)
print("✅ TESTES CONCLUÍDOS")
print("=" * 60)
'''

with open('test_api.py', 'w', encoding='utf-8') as f:
    f.write(test_code)

print("✅ test_api.py criado")
import requests

BASE_URL = "http://127.0.0.1:8000/api/v1/chat/query"
HEADERS = {"Content-Type": "application/json"}

# Simulamos el comportamiento del frontend: mismo thread_id para todas las consultas
thread_id = "frontend-session-123"

print("=" * 60)
print("SIMULACION DEL FRONTEND - Mismo thread_id para todas las consultas")
print("=" * 60)

# Consulta 1
print("\n--- CONSULTA 1 ---")
q1 = "¿Cómo configurar una VNet en Azure?"
r1 = requests.post(BASE_URL, json={"text": q1, "thread_id": thread_id}, headers=HEADERS)
if r1.status_code == 200:
    data1 = r1.json()
    print(f"Pregunta: {q1}")
    print(f"Respuesta: {data1['response'][:100]}...")
    print(f"Thread ID: {data1['thread_id']}")
    print(f"Fuentes: {[s.get('title') for s in data1.get('sources', [])]}")

# Consulta 2 - Diferente pregunta
print("\n--- CONSULTA 2 (DIFERENTE PREGUNTA) ---")
q2 = "¿Cómo aseguro una base de datos SQL en Azure?"
r2 = requests.post(BASE_URL, json={"text": q2, "thread_id": thread_id}, headers=HEADERS)
if r2.status_code == 200:
    data2 = r2.json()
    print(f"Pregunta: {q2}")
    print(f"Respuesta: {data2['response'][:100]}...")
    print(f"Thread ID: {data2['thread_id']}")
    print(f"Fuentes: {[s.get('title') for s in data2.get('sources', [])]}")

# Validación
print("\n" + "=" * 60)
print("VALIDACION")
print("=" * 60)
if r1.status_code == 200 and r2.status_code == 200:
    resp1 = data1['response'][:100]
    resp2 = data2['response'][:100]
    if resp1 == resp2:
        print("ERROR - Las respuestas son IDENTICAS (primeros 100 caracteres)")
        print(f"Respuesta 1: {resp1}")
        print(f"Respuesta 2: {resp2}")
    else:
        print("OK - Las respuestas son DIFERENTES")
        print(f"Respuesta 1: {resp1}")
        print(f"Respuesta 2: {resp2}")

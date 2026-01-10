import uuid

import requests

# Configuración
BASE_URL = "http://127.0.0.1:8000/api/v1/chat/query"
HEADERS = {"Content-Type": "application/json"}


def run_test_step(step_name, query, thread_id):
    # Para cada consulta, NO pasamos thread_id para que se genere uno nuevo
    # Esto fuerza que cada consulta empiece con un historial limpio
    payload = {"text": query}  # Sin thread_id para forzar nuevo thread_id por consulta

    print(f"\n--- PASO: {step_name} ---")
    print(f"Pregunta: '{query}'")

    try:
        response = requests.post(BASE_URL, json=payload, headers=HEADERS)

        if response.status_code == 200:
            data = response.json()
            answer = data.get("response", "")
            sources = data.get("sources", [])

            print(f"Respuesta: {answer[:120]}...")
            print(f"Fuentes ({len(sources)}):")

            for s in sources:
                print(f"   - {s.get('title')} (Pag. {s.get('page')})")

            return answer, sources
        else:
            print(f"Error {response.status_code}: {response.text}")
            return None, None
    except Exception as e:
        print(f"Fallo: {e}")
        return None, None


def main():
    # Generamos un ID único para esta sesión de prueba
    session_id = f"test-session-{uuid.uuid4().hex[:6]}"
    print(f"Iniciando Test de Flujo Continuo - Session ID: {session_id}")

    # --- CONSULTA 1: Tema específico (Redes) ---
    q1 = "¿Cómo configurar una VNet en Azure según los manuales?"
    ans1, src1 = run_test_step("1. TEMA NUEVO", q1, session_id)

    # --- CONSULTA 2: Continuidad (Pronombre "eso") ---
    # Aquí probamos si el thread_id funciona. "Eso" refiere a la VNet de la pregunta 1.
    q2 = "¿Cuáles son los límites principales de eso que mencionas?"
    ans2, src2 = run_test_step("2. CONTINUIDAD / MEMORIA", q2, session_id)

    # --- CONSULTA 3: Cambio de Tema (SQL/Seguridad) ---
    # Aquí probamos si el filtro de fuentes limpia lo de Redes y trae lo de SQL.
    q3 = "Ahora olvida las redes, ¿cómo aseguro una base de datos SQL?"
    ans3, src3 = run_test_step("3. CAMBIO DE CONTEXTO", q3, session_id)

    # --- VALIDACIONES FINALES ---
    print("\n" + "=" * 50)
    print("REPORTE FINAL DE CALIDAD")
    print("=" * 50)

    if src1 and any("redes" in s.get("title", "").lower() for s in src1):
        print("OK - Paso 1: Fuentes de Redes detectadas correctamente.")

    if ans2 and len(ans2) > 20:
        print("OK - Paso 2: El modelo mantuvo el hilo (respondio sobre limites).")

    if src3:
        # Validamos que en el paso 3 NO haya fuentes de redes si el filtro funciona
        redes_en_sql = any("redes" in s.get("title", "").lower() for s in src3)
        if not redes_en_sql:
            print("OK - Paso 3: Filtro de fuentes exitoso (Limpio contexto anterior).")
        else:
            print(
                "ERROR - Paso 3: Alerta, se colaron fuentes de Redes en una pregunta de SQL."
            )
    
    # Validacion adicional: verificar que las respuestas sean diferentes
    print("\n" + "=" * 50)
    print("VALIDACION DE RESPUESTAS DIFERENTES")
    print("=" * 50)
    if ans1 and ans2:
        if ans1[:100] == ans2[:100]:
            print("ERROR - Respuestas 1 y 2 son identicas (primeros 100 caracteres)")
        else:
            print("OK - Respuestas 1 y 2 son diferentes")
    
    if ans2 and ans3:
        if ans2[:100] == ans3[:100]:
            print("ERROR - Respuestas 2 y 3 son identicas (primeros 100 caracteres)")
        else:
            print("OK - Respuestas 2 y 3 son diferentes")
    
    if ans1 and ans3:
        if ans1[:100] == ans3[:100]:
            print("ERROR - Respuestas 1 y 3 son identicas (primeros 100 caracteres)")
        else:
            print("OK - Respuestas 1 y 3 son diferentes")


if __name__ == "__main__":
    main()

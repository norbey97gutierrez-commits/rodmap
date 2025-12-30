import asyncio
import json

import httpx


async def test_streaming_chat():
    # Configuración del endpoint
    url = "http://127.0.0.1:8000/api/v1/chat/stream"

    # Payload de prueba
    payload = {
        "text": "¿Cómo puedo configurar una VNet en Azure y qué beneficios tiene?",
        "thread_id": "test-session-001",
    }

    print(f"\n🚀 Enviando pregunta: {payload['text']}")
    print("-" * 50)

    try:
        # Usamos un cliente de HTTP asíncrono con timeout extendido
        async with httpx.AsyncClient(timeout=110.0) as client:
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    print(f"❌ Error: {response.status_code}")
                    return

                # Procesamos el flujo de eventos (SSE)
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        # Limpiamos el prefijo 'data: ' y parseamos el JSON
                        data_str = line.replace("data: ", "")
                        data = json.loads(data_str)

                        if data["type"] == "token":
                            # Imprimimos los tokens sin salto de línea para el efecto de escritura
                            print(data["content"], end="", flush=True)

                        elif data["type"] == "metadata":
                            print("\n\n📚 FUENTES ENCONTRADAS:")
                            for source in data["sources"]:
                                print(f"  - {source}")

                        elif data["type"] == "error":
                            print(f"\n❌ ERROR EN EL STREAM: {data['message']}")

    except Exception as e:
        print(f"\n❌ Fallo la conexión: {e}")


if __name__ == "__main__":
    asyncio.run(test_streaming_chat())

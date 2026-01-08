import asyncio

from src.api.graph import app


async def test_chat():
    pregunta = "¿Cómo se configura una VNet en Azure según la documentación cargada?"
    # O si tus documentos son de otro tema, usa una palabra clave de ellos.

    # Probamos enviando tanto 'question' como 'input' para ver cuál acepta tu Grafo
    inputs = {"question": pregunta, "input": pregunta}

    config = {"configurable": {"thread_id": "test-session-001"}}

    print(f"\n🤔 Preguntando: {pregunta}\n")

    try:
        async for event in app.astream(inputs, config=config):
            # Imprimimos el evento completo para ver qué está pasando dentro
            for node, data in event.items():
                print(f"\n--- Nodo ejecutado: {node} ---")
                print(f"Contenido del nodo: {data}")

                if "answer" in data:
                    print(f"\n🤖 RESPUESTA FINAL: {data['answer']}")

    except Exception:
        # Esto nos dará más detalle si vuelve a fallar
        import traceback

        print("❌ Error detallado:")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_chat())

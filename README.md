# 🚀 Azure AI Roadmap - Backend (Caso 8)

Este proyecto representa un Ingeniero de IA en Azure. Es un sistema de asistencia técnica profesional que utiliza arquitecturas de **Agentes RAG** (Retrieval-Augmented Generation) para responder consultas basadas en documentación específica.

## 🏗️ Arquitectura del Sistema

El backend está diseñado siguiendo una evolución natural de capacidades de IA, desde la inferencia mínima hasta la orquestación compleja de agentes:

* **API Layer**: FastAPI local para exposición de endpoints HTTP.
* **Orquestación**: LangGraph para la gestión de flujos, estados y control de ciclos.
* **Razonamiento**: Azure OpenAI con **Structured Output** para clasificación de intenciones mediante Pydantic.
* **Conocimiento**: Azure AI Search para la indexación y búsqueda semántica de documentos técnicos.

## 🚀 Configuración e Instalación

### 1. Requisitos Previos

* Python 3.10+
* Cuenta de Azure con recursos de **Azure OpenAI** y **Azure AI Search**[cite: 5, 6].

### 2. Variables de Entorno (`.env`)

* Crea un archivo `.env` en la raíz del backend con los siguientes parámetros:

    AZURE_OPENAI_API_KEY="tu_llave"
    AZURE_OPENAI_ENDPOINT="tu_endpoint"
    AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4o-mini"
    AZURE_AI_SEARCH_SERVICE_NAME="tu_servicio"
    AZURE_AI_SEARCH_INDEX_NAME="tu_indice"
    AZURE_AI_SEARCH_API_KEY="tu_api_key"

### 3. Instalación de Dependencias

    pip install -r requirements.txt

### 4. Ejecución del Servidor

     uvicorn src.main:app --reload

Acceda a la documentación interactiva Swagger en:<http://127.0.0.1:8000/docs>

### 📡 Puntos finales principales

GET /health: Verificación de estado del servicio.

POST /api/v1/chat/stream: Endpoint principal que recibe la pregunta y devuelve un JSON estructurado con la respuesta y las fuentes.

### 📂 Estructura del Proyecto

La arquitectura propuesta es:

    RODMAP/
    ├── data/               # Archivos JSON locales para indexación
    ├── src/
    │   ├── api/
    │   │   ├── core/       # Configuración (.env), estados y modelos Pydantic
    │   │   ├── llm/        # Clientes de Azure OpenAI y lógica del Clasificador
    │   │   ├── search/     # Utilidades para búsqueda en Azure AI Search
    │   │   ├── graph.py    # Definición del flujo de LangGraph (Nodos y Edges)
    │   │   └── routes.py   # Endpoints de FastAPI (POST /query)
    │   ├── tools/          # Herramientas del agente (indexer.py)
    │   └── main.py         # Punto de entrada de la aplicación FastAPI
    ├── .env                # Variables de entorno (Azure Keys & Endpoints)
    └── requirements.txt    # Dependencias del proyecto

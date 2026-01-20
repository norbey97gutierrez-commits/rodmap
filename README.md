# 🚀 Arquitecto Azure AI - Backend

Este proyecto representa un Ingeniero de IA en Azure. Es un sistema de asistencia técnica profesional que utiliza arquitecturas de **Agentes RAG** (Retrieval-Augmented Generation) para responder consultas basadas en documentación específica.

## 🏗️ Arquitectura del Sistema

El backend está diseñado siguiendo una evolución natural de capacidades de IA, desde la inferencia mínima hasta la orquestación compleja de agentes:

* **API Layer**: FastAPI para exposición de endpoints HTTP.
* **Orquestación**: LangGraph para la gestión de flujos, estados y control de ciclos.
* **Razonamiento**: Azure OpenAI con **Structured Output** para clasificación de intenciones.
* **Conocimiento**: Azure AI Search para indexación y búsqueda semántica.
* **Persistencia**: PostgreSQL para usuarios y `kv_store`.

## 🚀 Configuración e Instalación

### 1. Requisitos Previos

* Python 3.12+
* Cuenta de Azure con recursos de **Azure OpenAI** y **Azure AI Search**.

### 2. Variables de Entorno (`.env`)

* Crea un archivo `.env` en la raíz del backend con los siguientes parámetros:

```sh
AZURE_OPENAI_API_KEY="tu_llave"
AZURE_OPENAI_ENDPOINT="tu_endpoint"
AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4o-mini"
AZURE_AI_SEARCH_SERVICE_NAME="tu_servicio"
AZURE_AI_SEARCH_INDEX_NAME="tu_indice"
AZURE_AI_SEARCH_API_KEY="tu_api_key"
FRONTEND_URL="http://localhost:5173"
DATABASE_URL="tu configuracion"
```

En Docker, `DATABASE_URL` se define automáticamente en `docker-compose.yml`.

### 3. Instalación de Dependencias (uv recomendado)

```sh
uv sync
```

### 4. Ejecución del Servidor

```sh
uv run uvicorn src.main:app --reload
```

### 5. Docker (PostgreSQL + API)

```sh
docker compose up --build
```

La API queda disponible en `http://127.0.0.1:8000` y Postgres en `localhost:5432`.
El frontend (React+Vite) se sirve en `http://localhost:5173/chat`.

Acceda a la documentación interactiva Swagger en:<http://127.0.0.1:8000/docs>

### 📡 Puntos finales principales

GET /health: Verificación de estado del servicio.

POST /api/v1/chat/stream: Endpoint principal que recibe la pregunta y devuelve un JSON estructurado con la respuesta y las fuentes.

### 📂 Estructura del Proyecto

La arquitectura propuesta es:

```sh
    RODMAP/
    ├── data/               # Archivos JSON locales para indexación
    ├── src/
    │   ├── adapters/       # Adaptadores (Azure, local, parsers)
    │   ├── application/    # Orquestación LangGraph y estado
    │   ├── domain/         # Entidades y puertos
    │   ├── infrastructure/ # Configuración y seguridad
    │   ├── routes/         # Endpoints FastAPI
    │   └── main.py         # Punto de entrada de la aplicación
    ├── .env                # Variables de entorno (Azure Keys & Endpoints)
    ├── docker-compose.yml  # API + PostgreSQL
    └── pyproject.toml      # Dependencias del proyecto (uv)

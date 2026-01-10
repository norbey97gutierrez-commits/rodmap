# 📊 Diagrama de Flujo Completo del Sistema - Azure AI Architect Backend

Este documento contiene el diagrama de flujo completo del sistema, incluyendo todos los archivos y sus interacciones.

## 🔄 Flujo Principal del Sistema

```mermaid
flowchart TB
    %% ============================================
    %% CAPA DE ENTRADA - FASTAPI
    %% ============================================
    Start([Cliente/Frontend]) --> Main[main.py<br/>FastAPI App]
    Main --> CORS[CORS Middleware<br/>Permite CORS]
    CORS --> Routes[routes.py<br/>Router API]
    
    %% ============================================
    %% ENDPOINTS
    %% ============================================
    Routes --> Health[GET /health<br/>Health Check]
    Routes --> Query[POST /api/v1/chat/query<br/>Chat Endpoint]
    
    %% ============================================
    %% PROCESAMIENTO DE QUERY
    %% ============================================
    Query --> ThreadID{Thread ID<br/>existe?}
    ThreadID -->|No| NewThread[Generar nuevo<br/>UUID Thread ID]
    ThreadID -->|Sí| CheckState[Verificar estado<br/>del checkpointer]
    CheckState --> InputChanged{Input<br/>cambió?}
    InputChanged -->|Sí| NewThread
    InputChanged -->|No| UseThread[Usar Thread ID<br/>existente]
    NewThread --> UseThread
    
    %% ============================================
    %% LANGGRAPH - GRAFO DE ORQUESTACIÓN
    %% ============================================
    UseThread --> Graph[graph.py<br/>LangGraph App]
    Graph --> Checkpointer[MemorySaver<br/>Checkpointer]
    
    %% ============================================
    %% NODO ROUTER - PUNTO DE ENTRADA DEL GRAFO
    %% ============================================
    Graph --> Router[router_node<br/>Clasificación de Intención]
    Router --> Classifier[classifier.py<br/>classify_intent]
    Classifier --> LLMClient[client.py<br/>AzureChatOpenAI]
    LLMClient --> Config[config.py<br/>Settings]
    Config --> Env[.env<br/>Variables de Entorno]
    
    Classifier --> IntentionResponse[IntentionResponse<br/>SALUDO / PREGUNTA_TECNICA<br/>/ FUERA_DE_DOMINIO]
    
    %% ============================================
    %% RUTEO CONDICIONAL DESPUÉS DE CLASIFICACIÓN
    %% ============================================
    IntentionResponse --> RouteDecision{route_after_classifier}
    RouteDecision -->|FUERA_DE_DOMINIO| OutOfDomain[out_of_domain_node<br/>Respuesta fuera de dominio]
    RouteDecision -->|SALUDO o<br/>PREGUNTA_TECNICA| Agent[agent_node<br/>Generación de Respuesta]
    
    %% ============================================
    %% NODO AGENT - GENERACIÓN CON LLM
    %% ============================================
    Agent --> State[state.py<br/>GraphState]
    State --> MergeHistory[merge_history_with_reset<br/>Gestión de Historial]
    MergeHistory --> ValidateHistory[_validate_and_filter_history<br/>Validar ToolMessages]
    
    Agent --> SystemPrompt[SystemMessage<br/>Prompt del Sistema]
    SystemPrompt --> LLMInvoke[LLM.ainvoke<br/>con herramientas]
    LLMInvoke --> LLMClient
    
    %% ============================================
    %% DECISIÓN: ¿NECESITA HERRAMIENTAS?
    %% ============================================
    LLMInvoke --> HasToolCalls{should_continue<br/>¿Tiene tool_calls?}
    HasToolCalls -->|Sí| Tools[custom_tool_wrapper<br/>Ejecución de Herramientas]
    HasToolCalls -->|No| Finalize[finalize_node<br/>Extracción de Respuesta]
    
    %% ============================================
    %% NODO TOOLS - BÚSQUEDA EN AZURE AI SEARCH
    %% ============================================
    Tools --> SearchTool[search_technical_docs<br/>StructuredTool]
    SearchTool --> SearchService[service.py<br/>AzureAISearchService]
    
    SearchService --> Embeddings[AzureOpenAIEmbeddings<br/>Generación de Vectores]
    Embeddings --> Config
    Embeddings --> EmbedQuery[Embed Query<br/>Convertir texto a vector]
    
    SearchService --> SearchClient[Azure SearchClient<br/>Búsqueda Híbrida]
    SearchClient --> VectorQuery[VectorizedQuery<br/>Búsqueda Vectorial]
    SearchClient --> TextQuery[Búsqueda por Texto<br/>Búsqueda de Palabras Clave]
    
    VectorQuery --> HybridSearch[Búsqueda Híbrida<br/>Vectorial + Texto]
    TextQuery --> HybridSearch
    
    HybridSearch --> AzureSearch[Azure AI Search<br/>Índice Vectorial]
    AzureSearch --> Documents[data/documents.json<br/>Documentos Indexados]
    
    SearchService --> FormatResults[Formatear Resultados<br/>content + value]
    FormatResults --> ToolMessage[ToolMessage<br/>Resultado de Búsqueda]
    
    %% ============================================
    %% RETORNO AL AGENT DESPUÉS DE TOOLS
    %% ============================================
    ToolMessage --> Agent
    
    %% ============================================
    %% NODO FINALIZE - EXTRACCIÓN DE RESPUESTA
    %% ============================================
    Finalize --> ExtractResponse[Extraer último<br/>AIMessage sin tool_calls]
    ExtractResponse --> ExtractSources[Extraer fuentes<br/>del último turno]
    ExtractSources --> FilterSources[Filtrar fuentes<br/>por relevancia]
    
    %% ============================================
    %% RESPUESTA FINAL
    %% ============================================
    FilterSources --> Response[ChatResponse<br/>thread_id, intention,<br/>response, sources, status]
    OutOfDomain --> Response
    
    Response --> Routes
    Routes --> Main
    Main --> End([Respuesta JSON<br/>al Cliente])
    
    %% ============================================
    %% HERRAMIENTAS DE INDEXACIÓN
    %% ============================================
    Indexer[indexer.py<br/>index_documents] --> IndexService[AzureAISearchService]
    IndexService --> CreateIndex[create_or_update_index<br/>Crear/Actualizar Índice]
    CreateIndex --> IndexSchema[Definir Esquema<br/>Campos + Vector Search]
    
    Indexer --> LoadDocs[Cargar documents.json]
    LoadDocs --> Documents
    LoadDocs --> ProcessDocs[Procesar Documentos<br/>Generar Embeddings]
    ProcessDocs --> Embeddings
    
    ProcessDocs --> UpsertVectors[upsert_vectors<br/>Subir Vectores a Azure]
    UpsertVectors --> AzureSearch
    
    %% ============================================
    %% SCRIPT DE RESET
    %% ============================================
    Reset[reset_index.py<br/>reset] --> ResetService[AzureAISearchService]
    ResetService --> DeleteIndex[delete_index<br/>Eliminar Índice]
    DeleteIndex --> AzureSearch
    
    %% ============================================
    %% ARCHIVOS DE PRUEBA
    %% ============================================
    TestChat[test_chat.py<br/>Test Grafo Directo] --> Graph
    TestRAG[test_rag.py<br/>Test RAG Simple] --> Retriever[retriever.py<br/>rag_query]
    TestRAGSources[test_rag_sources.py<br/>Test Fuentes] --> Query
    TestFrontend[test_frontend_simulation.py<br/>Simulación Frontend] --> Query
    
    Retriever --> VectorStore[AzureSearch<br/>Vector Store]
    VectorStore --> AzureSearch
    
    %% ============================================
    %% ESTILOS
    %% ============================================
    classDef entryPoint fill:#e1f5ff,stroke:#01579b,stroke-width:3px
    classDef api fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef graph fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef llm fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
    classDef search fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    classDef data fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef tool fill:#e0f2f1,stroke:#004d40,stroke-width:2px
    classDef test fill:#f1f8e9,stroke:#33691e,stroke-width:2px
    
    class Start,End entryPoint
    class Main,CORS,Routes,Query,Health api
    class Graph,Router,Agent,Tools,Finalize,OutOfDomain,State,MergeHistory,ValidateHistory graph
    class LLMClient,Classifier,LLMInvoke,SystemPrompt,Embeddings,EmbedQuery llm
    class SearchService,SearchClient,VectorQuery,TextQuery,HybridSearch,AzureSearch,VectorStore,SearchTool search
    class Documents,Env,Config data
    class Indexer,Reset,CreateIndex,ProcessDocs,UpsertVectors,DeleteIndex tool
    class TestChat,TestRAG,TestRAGSources,TestFrontend test
```

## 📁 Estructura de Archivos y Responsabilidades

### 🚀 **Capa de Entrada (API)**
- **`main.py`**: Punto de entrada FastAPI, configuración CORS, registro de rutas
- **`routes.py`**: Endpoints HTTP (`/api/v1/chat/query`, `/health`), manejo de thread_id, invocación del grafo

### 🧠 **Capa de Orquestación (LangGraph)**
- **`graph.py`**: Define el grafo de estado con nodos (router, agent, tools, finalize, out_of_domain) y lógica de control
- **`state.py`**: Define `GraphState` (TypedDict) y funciones de merge para historial y fuentes

### 🤖 **Capa de LLM**
- **`llm/client.py`**: Cliente AzureChatOpenAI (configuración, inicialización)
- **`llm/classifier.py`**: Clasificador de intenciones usando Structured Output (SALUDO, PREGUNTA_TECNICA, FUERA_DE_DOMINIO)

### 🔍 **Capa de Búsqueda**
- **`search/service.py`**: `AzureAISearchService` - búsqueda híbrida (vectorial + texto), gestión de índices
- **`search/retriever.py`**: Retriever RAG simple (no usado en el flujo principal, solo para tests)

### ⚙️ **Capa de Configuración**
- **`core/config.py`**: Settings (Pydantic) - carga variables de entorno, validación
- **`core/state.py`**: Estado del grafo y funciones de merge

### 🛠️ **Herramientas**
- **`tools/indexer.py`**: Script de indexación - carga `documents.json`, genera embeddings, sube a Azure AI Search
- **`reset_index.py`**: Script para eliminar y recrear el índice

### 📊 **Datos**
- **`data/documents.json`**: Documentos técnicos en formato JSON con metadatos (id, title, content, source, page_number)

### 🧪 **Tests**
- **`test_chat.py`**: Test directo del grafo LangGraph
- **`test_rag.py`**: Test del retriever RAG simple
- **`test_rag_sources.py`**: Test del endpoint completo con validación de fuentes
- **`test_frontend_simulation.py`**: Simulación del comportamiento del frontend

## 🔄 Flujo Detallado de una Consulta

### 1. **Recepción de Request** (`routes.py`)
   - Cliente envía POST a `/api/v1/chat/query` con `text` y opcionalmente `thread_id`
   - Si no hay `thread_id`, se genera uno nuevo
   - Si hay `thread_id`, se verifica si el input cambió (comparando con estado del checkpointer)
   - Si el input cambió, se genera un nuevo `thread_id` para reiniciar el historial

### 2. **Invocación del Grafo** (`graph.py`)
   - Se invoca `graph_app.ainvoke()` con el input y config (thread_id)
   - El checkpointer (MemorySaver) restaura el estado previo si existe

### 3. **Nodo Router** (`router_node`)
   - Clasifica la intención usando `classify_intent()` (llama a `classifier.py`)
   - `classifier.py` usa Structured Output con Azure OpenAI para clasificar
   - Retorna `intention` y prepara el historial con el nuevo `HumanMessage`

### 4. **Ruteo Condicional** (`route_after_classifier`)
   - Si `intention == "FUERA_DE_DOMINIO"` → va a `out_of_domain_node`
   - Si no → va a `agent_node`

### 5. **Nodo Agent** (`agent_node`)
   - Valida y filtra el historial usando `_validate_and_filter_history()`
   - Asegura que cada `AIMessage` con `tool_calls` tenga sus `ToolMessages` inmediatamente después
   - Construye el prompt del sistema enfatizando responder solo a la pregunta actual
   - Invoca el LLM con herramientas usando `llm.bind_tools(tools).ainvoke()`
   - El LLM puede decidir llamar a `search_technical_docs` o responder directamente

### 6. **Decisión de Continuación** (`should_continue`)
   - Si el último mensaje tiene `tool_calls` → va a `custom_tool_wrapper`
   - Si no → va a `finalize_node`

### 7. **Nodo Tools** (`custom_tool_wrapper`)
   - Ejecuta cada herramienta (actualmente solo `search_technical_docs`)
   - `search_technical_docs`:
     - Genera embeddings del query usando `AzureOpenAIEmbeddings`
     - Ejecuta búsqueda híbrida (vectorial + texto) en Azure AI Search
     - Retorna resultados formateados como `{"content": "...", "value": [...]}`
   - Crea `ToolMessage` para cada `tool_call`
   - Retorna al `agent_node` para que el LLM procese los resultados

### 8. **Nodo Finalize** (`finalize_node`)
   - Extrae el último `AIMessage` sin `tool_calls` (la respuesta final)
   - Extrae fuentes solo del último turno de conversación (desde el último `HumanMessage`)
   - Filtra fuentes por relevancia (solo las que aparecen mencionadas en la respuesta)
   - Retorna `response` y `sources`

### 9. **Respuesta Final** (`routes.py`)
   - Construye `ChatResponse` con `thread_id`, `intention`, `response`, `sources`, `status`
   - Retorna JSON al cliente

## 🔄 Gestión del Historial

El sistema usa `merge_history_with_reset` para gestionar el historial:

1. **Detección de Cambio de Input**: Compara el último `HumanMessage` en el historial existente con el nuevo
2. **Reinicio Inteligente**: Si el input cambió Y no hay `tool_calls` pendientes → reinicia el historial completamente
3. **Validación de ToolMessages**: `_validate_and_filter_history` asegura que cada `AIMessage` con `tool_calls` tenga sus `ToolMessages` inmediatamente después (requisito de Azure OpenAI)

## 🔍 Búsqueda Híbrida

1. **Generación de Embeddings**: El query se convierte en un vector usando `AzureOpenAIEmbeddings`
2. **Búsqueda Vectorial**: Se busca en el índice usando `VectorizedQuery` (K=5 vecinos más cercanos)
3. **Búsqueda de Texto**: Se busca por palabras clave usando `search_text`
4. **Combinación**: Azure AI Search combina ambos resultados (búsqueda híbrida)
5. **Formateo**: Los resultados se formatean con `content` (para el LLM) y `value` (para las fuentes)

## 📦 Indexación de Documentos

1. **Carga**: Se carga `data/documents.json`
2. **Procesamiento**: Para cada documento:
   - Se genera un embedding del texto (título + contenido)
   - Se estructura el documento con metadatos (id, title, content, content_vector, source, page_number)
3. **Creación de Índice**: Se crea/actualiza el índice en Azure AI Search con el esquema definido
4. **Subida**: Se suben los documentos vectorizados usando `upsert_vectors`

## 🎯 Puntos Clave del Sistema

- **Thread Management**: Cada consulta puede tener su propio `thread_id` o continuar una conversación
- **Historial Inteligente**: Se reinicia automáticamente cuando el input cambia
- **Validación Robusta**: Garantiza que los `ToolMessages` estén correctamente asociados a sus `tool_calls`
- **Búsqueda Híbrida**: Combina búsqueda semántica (vectorial) con búsqueda de palabras clave
- **Structured Output**: Usa Pydantic para clasificación de intenciones garantizada
- **Manejo de Errores**: Cada componente tiene manejo robusto de errores con mensajes informativos

from fastapi import FastAPI
from app.application.services.document_chunking_service import DocumentChunkingService
from app.application.services.embedding_ingest_service import EmbeddingIngestService
from app.application.services.file_storage_service import FileStorageService
from app.application.services.graph_ingest_service import GraphIngestService
from app.application.services.graph_query_service import GraphQueryService
from app.application.usecases.extract_graph_usecase import ExtractGraphUseCase
from app.application.usecases.upload_usecase import UploadUseCase
from app.config.paths import UPLOAD_DIR
from app.core.graph.graph_store import GraphStore
from app.infrastructure.models.model_loader import ModelRegistry
from app.routes import upload
from app.routes import graph
from app.routes import inference
from app.globals import get_registry, set_registry
from app.infrastructure.models.model_provider import ModelProvider
from app.application.services.retrieval_service import RetrievalService
from app.application.services.answer_generation_service import AnswerGenerationService
from app.application.services.graph_extraction_service import GraphExtractionService
from app.application.usecases.ask_question_usecase import AskQuestionUseCase

import torch, gc

app = FastAPI(
    title="GraphRAG Explorer API",
    description="Day 2: Upload & Chunk",
    version="0.2"
)

app.include_router(upload.router, prefix="/api")
app.include_router(graph.router)
app.include_router(inference.router)

@app.get("/")
def root():
    return {"status": "GraphRAG Explorer backend running"}

@app.get("/health")
def health_check():
    return {"ok": True}

@app.on_event("startup")
async def load_models():
    registry = ModelRegistry()
    provider = ModelProvider(registry)

    # 方案1: 採用FastAPI 的注入功能
    app.state.registry = registry
    app.state.model_provider = provider

    # 方案2: 自訂義, 全域取用 (全域可取，不必傳 request)
    set_registry(registry)

    # 在main組好service
    graph_store = GraphStore()
    
    graph_ingest_service = GraphIngestService(
        provider=provider,
        store=graph_store,
    )

    graph_query_service = GraphQueryService(
        store=graph_store,
    )

    extract_graph_usecase = ExtractGraphUseCase(
        ingest_service=graph_ingest_service
    )

    # === Application Services ===
    retrieval_service = RetrievalService(provider=provider)
    answer_generation_service = AnswerGenerationService(provider=provider)
    graph_extraction_service = GraphExtractionService(provider=provider)
    file_storage_service = FileStorageService(UPLOAD_DIR)
    document_chunker_service = DocumentChunkingService()
    embedding_ingestor_service = EmbeddingIngestService(registry)

    # === UseCase ===
    ask_question_usecase = AskQuestionUseCase(
        retrieval=retrieval_service,
        answer_generator=answer_generation_service,
        graph_extractor=graph_extraction_service,
    )
    
    upload_usecase = UploadUseCase(
        chunker=document_chunker_service,
        ingestor=embedding_ingestor_service,
    )

    # 掛到 app.state
    app.state.ask_question_usecase = ask_question_usecase
    app.state.upload_usecase = upload_usecase
    app.state.file_storage_service = file_storage_service
    
    app.state.graph_store = graph_store
    app.state.graph_ingest_service = graph_ingest_service
    app.state.graph_query_service = graph_query_service
    app.state.extract_graph_usecase = extract_graph_usecase

    print("⚙️ ModelRegistry ready (lazy mode, 尚未載入模型)")

@app.on_event("shutdown")
def release_gpu():
    print("🧹 Releasing GPU memory before shutdown...")

    registry = get_registry()
    if registry:
        registry.unload_all()
        
    gc.collect()
    torch.cuda.empty_cache()
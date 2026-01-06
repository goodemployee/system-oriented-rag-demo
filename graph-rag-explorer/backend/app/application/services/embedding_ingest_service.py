from app.infrastructure.models.model_loader import ModelRegistry


class EmbeddingIngestService:
    def __init__(self, registry: ModelRegistry):
        self._registry = registry

    def ingest(self, texts: list[str]) -> None:
        """
        將文本加入向量資料庫
        """
        self._registry.add_chunks(texts)

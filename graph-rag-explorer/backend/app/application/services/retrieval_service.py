from app.infrastructure.models.model_provider import ModelProvider


class RetrievalService:
    def __init__(self, provider: ModelProvider):
        self.provider = provider
        
    def retrieve(self, query: str) -> list[str]:
        embedder = self.provider.get_embedder()
        docs = embedder.query(query)
        return [d["text"] for d in docs]

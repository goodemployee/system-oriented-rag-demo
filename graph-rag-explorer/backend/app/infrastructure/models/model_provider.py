# app/infrastructure/models/model_provider.py

from app.infrastructure.models.model_loader import ModelRegistry
from app.core.llm.llm import LLM
from app.core.embedding.embedder import Embedder

class ModelProvider:
    """
    模型提供者（Facade）

    職責：
    - 對應用層提供「可用的模型能力」
    - 不負責模型初始化策略
    - 不暴露 model_id / device / config

    依賴：
    - ModelRegistry 作為 runtime / lifecycle 管理者
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry

    # === 對外提供能力 ===

    def get_llm(self) -> LLM:
        """
        取得可用的 LLM（已初始化或 lazy 載入）
        """
        return self._registry._get_llm_internal()

    def get_embedder(self) -> Embedder:
        """
        取得可用的 Embedder
        """
        return self._registry._get_embedder_internal()

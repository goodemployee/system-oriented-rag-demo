from __future__ import annotations

from typing import Dict, List, Optional

from app.core.graph.graph_store import GraphStore
from app.core.graph.graph_builder import GraphBuilder
from app.infrastructure.models.model_provider import ModelProvider


class GraphIngestService:
    """
    GraphIngestService 是應用層（Application Service），
    負責協調模型、圖譜建構流程，對外提供「資料匯入」能力。

    本類別不處理實際抽取邏輯，也不關心圖譜儲存細節，
    僅負責 orchestration。
    """

    def __init__(
        self,
        provider: ModelProvider,
        store: GraphStore,
    ) -> None:
        """
        建立 GraphIngestService。

        Args:
            provider: 提供 LLM / embedding 等模型資源的 ModelProvider。
            store: 知識圖譜儲存層。
        """
        self.provider: ModelProvider = provider
        self.store: GraphStore = store

    def ingest_from_file(
        self,
        file_path: str,
        *,
        max_chunks: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """
        從檔案匯入資料並建立知識圖譜。

        此方法會：
        - 建立 GraphBuilder
        - 呼叫其 build_from_file
        - 回傳本次匯入所產生的三元組

        Args:
            file_path: 文件路徑。
            max_chunks: 最大處理 chunk 數量，None 表示不限制。

        Returns:
            本次匯入產生的三元組清單。
        """
        builder: GraphBuilder = GraphBuilder(
            provider=self.provider,
            store=self.store,
        )

        return builder.build_from_file(
            file_path=file_path,
            max_chunks=max_chunks,
        )

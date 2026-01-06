from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from app.application.services.document_chunking_service import DocumentChunkingService
from app.application.services.embedding_ingest_service import EmbeddingIngestService
from app.core.embedding.chunker import DocumentChunk


# ⚠️ 注意
# UseCase 不應該直接從 app.state 拿 provider
# 只讓「組裝層」有地方拿。

class UploadUseCase:
    """
    UploadUseCase 是 Upload 流程的 UseCase / Orchestrator。

    職責：
    - 協調文件切分
    - 協調 embedding ingest
    - 組合並回傳開發期結果
    """

    def __init__(
        self,
        chunker: DocumentChunkingService,
        ingestor: EmbeddingIngestService,
    ) -> None:
        self.chunker: DocumentChunkingService = chunker
        self.ingestor: EmbeddingIngestService = ingestor

    async def execute(self, file_path: Path) -> Dict[str, object]:
        """
        執行 Upload 的完整流程。

        流程：
        1.（預留）存檔
        2. 切分文件為 chunks
        3. 將 chunk text 送入 embedding ingest
        4. 回傳開發期使用的結果資訊

        Args:
            file_path: 已存放完成的文件路徑。

        Returns:
            包含檔名、chunk 數量與 chunk 內容的結果 dict。
        """
        # 1️⃣ 存檔（目前由外部處理）

        # 2️⃣ 切 chunk
        chunks: List[DocumentChunk] = self.chunker.split(file_path)

        texts: List[str] = [c["text"] for c in chunks]

        # 3️⃣ 向量化
        self.ingestor.ingest(texts)

        # 4️⃣ 回傳（開發期 API）
        return {
            "filename": str(file_path),
            "chunks_stored": len(texts),
            "message": "文件分割並已加入向量資料庫",
            "chunks": chunks,
        }

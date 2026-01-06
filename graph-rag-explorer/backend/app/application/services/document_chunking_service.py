from __future__ import annotations

from pathlib import Path
from typing import List

from app.core.embedding.chunker import split_document, DocumentChunk


class DocumentChunkingService:
    """
    DocumentChunkingService 負責文件切分（chunking），
    作為 Application Service 封裝 chunker 的使用方式。
    """

    def split(self, file_path: Path) -> List[DocumentChunk]:
        """
        將指定檔案切分為 DocumentChunk 清單。

        Args:
            file_path: 文件路徑。

        Returns:
            文件切分後的 DocumentChunk 清單。
        """
        return split_document(str(file_path))

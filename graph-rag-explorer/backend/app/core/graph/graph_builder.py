from __future__ import annotations

import hashlib
from typing import Dict, List, Optional

from app.core.graph.graph_extractor import GraphExtractor
from app.core.graph.graph_store import GraphStore
from app.core.embedding.chunker import split_document, DocumentChunk
from app.infrastructure.models.model_provider import ModelProvider


def _h(s: str) -> str:
    """
    計算字串的 MD5 雜湊值，用於內容去重。

    Args:
        s: 原始字串內容。

    Returns:
        對應的 MD5 hex digest 字串。
    """
    return hashlib.md5(s.encode("utf-8")).hexdigest()


class GraphBuilder:
    """
    GraphBuilder 負責將文件轉換為知識圖譜（三元組）並寫入 GraphStore。

    職責包含：
    - 文件切 chunk
    - chunk 去重
    - 根據語意優先排序（關係句優先）
    - 呼叫 LLM 抽取三元組
    - 寫入圖譜儲存層
    """

    def __init__(
        self,
        provider: ModelProvider,
        store: GraphStore,
    ) -> None:
        """
        建立 GraphBuilder。

        Args:
            provider: 提供 LLM 的 ModelProvider。
            store: 用於儲存三元組的 GraphStore。
        """
        self.extractor: GraphExtractor = GraphExtractor(
            llm=provider.get_llm()
        )
        self.store: GraphStore = store

    def build_from_file(
        self,
        file_path: str,
        max_chunks: Optional[int] = 50,
    ) -> List[Dict[str, str]]:
        """
        從檔案建立知識圖譜三元組。

        流程：
        1. 切分文件為多個 chunk
        2. 依文字內容進行去重
        3. 將「看起來像關係句」的 chunk 排在前面
        4. 限制最大處理 chunk 數量
        5. 抽取三元組並寫入 GraphStore

        Args:
            file_path: 文件路徑。
            max_chunks: 最多處理的 chunk 數量（None 表示不限制）。

        Returns:
            從此檔案中抽取出的所有三元組清單。
        """
        chunks: List[DocumentChunk] = split_document(file_path)

        # 去重：hash -> text
        uniq: Dict[str, str] = {}
        for c in chunks:
            text: str = c["text"].strip()
            if not text:
                continue

            h: str = _h(text)
            if h not in uniq:
                uniq[h] = text

        texts: List[str] = list(uniq.values())

        # 關係句優先排序
        texts.sort(
            key=lambda t: 0 if self.extractor.looks_like_relation(t) else 1
        )

        if max_chunks is not None:
            texts = texts[:max_chunks]

        all_triples: List[Dict[str, str]] = []

        for t in texts:
            triples: List[Dict[str, str]] = self.extractor.extract_triples(t)
            if not triples:
                continue

            # 寫入三元圖
            self.store.add_triples(triples)
            all_triples.extend(triples)

        return all_triples

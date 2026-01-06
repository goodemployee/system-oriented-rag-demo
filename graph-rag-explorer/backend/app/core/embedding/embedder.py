# app/core/embedding/embedder.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, List, TypedDict

import torch
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.api.types import Documents, Embeddings
from app.config.paths import EMBEDDER_CACHE_DIR, CHROMA_DIR


class ChunkResult(TypedDict):
    text: str
    score: float


class Embedder:
    """
    Embedder
    ----------
    封裝 SentenceTransformer 模型 + ChromaDB 資料庫。
    用於：
      1. 新增文本段落（向量化並入庫）
      2. 根據 query 查詢最相似段落
    """

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        persist_dir: Optional[str] = None,
    ) -> None:
        self.model_id: str = model_id
        self.device: str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- cache 路徑 ---
        self.cache_dir = EMBEDDER_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # --- 向量資料庫 ---
        persist_dir = persist_dir or str(CHROMA_DIR)
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection("docs")

        # --- 模型本體 ---
        self.model: Optional[SentenceTransformer] = None
        print(f"🧩 Embedder 初始化完成 (model={self.model_id}, device={self.device})")

    # -------------------------------------------------------------------------
    # 模型載入與釋放
    # -------------------------------------------------------------------------
    def load(self) -> None:
        """載入 SentenceTransformer 模型"""
        if self.model:
            print("🔁 Embedder 已載入，略過。")
            return

        print(f"📦 正在載入 Embedder 模型：{self.model_id}")
        self.model = SentenceTransformer(
            self.model_id,
            device=self.device,
            cache_folder=str(self.cache_dir)
        )
        print("✅ Embedder 模型載入完成。")

    def unload(self) -> None:
        """釋放模型與 GPU 資源"""
        if self.model:
            del self.model
        self.model = None
        torch.cuda.empty_cache()
        print("✅ Embedder 已釋放。")

    # -------------------------------------------------------------------------
    # 文字向量化
    # -------------------------------------------------------------------------
    def embed(self, texts: list[str]) -> list[list[float]]:
        """將多段文字轉為向量"""
        if not self.model:
            self.load()

        if self.model == None:
            raise

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embeddings.tolist()

    # -------------------------------------------------------------------------
    # 新增資料到向量資料庫
    # -------------------------------------------------------------------------
    def add_chunks(self, texts: list[str]) -> None:
        """將多段文本向量化後存入 Chroma 資料庫"""
        if not texts:
            print("⚠️ add_chunks: 空文本列表，略過。")
            return

        if not self.model:
            self.load()

        print(f"🪣 新增 {len(texts)} 筆 chunk 至向量資料庫 ...")
        embeddings = self.embed(texts)
        ids = [f"chunk_{i}" for i in range(len(texts))]

        self.collection.add(
            documents=texts,
            embeddings=embeddings,  # type: ignore[arg-type]
            ids=ids
        )
        print("✅ 向量資料庫新增完成。")

    # -------------------------------------------------------------------------
    # 查詢相似文段
    # -------------------------------------------------------------------------
    def query(self, question: str, top_k: int = 5) -> List[ChunkResult]:
        """
        根據問題文字，查詢最相關的文段。
        回傳 [(text, score), ...]
        """
        if not self.model:
            self.load()

        query_vec = self.embed([question])[0]
        results = self.collection.query(
            query_embeddings=[query_vec],  # type: ignore[arg-type]
            n_results=top_k,
        )

        docs = results.get("documents", [[]])[0] # type: ignore
        scores = results.get("distances", [[]])[0] # type: ignore

        return [
            {"text": text, "score": float(score)}
            for text, score in zip(docs, scores)
        ]

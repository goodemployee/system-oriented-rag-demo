# app/config/modules.py
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ModulesConfig:
    """
    模組設定中樞（Single Source of Truth）

    只負責回答一件事：
    「系統現在用哪一個模組 / 模型？」
    """

    # --- LLM ---
    llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # --- Embedder ---
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Graph / Triple Extractor ---
    graph_extractor_model: str = "microsoft/Phi-3.5-mini-instruct"

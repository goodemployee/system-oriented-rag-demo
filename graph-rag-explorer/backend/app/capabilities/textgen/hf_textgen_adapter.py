from __future__ import annotations
from typing import Any
from app.capabilities.textgen.protocols import TextGenPipe
from app.capabilities.textgen.types import GenOutput

class HFTextGenAdapter(TextGenPipe):
    """將 Hugging Face text-generation pipeline 適配到 TextGenPipe 介面。"""
    def __init__(self, hf_pipeline: Any) -> None:
        self._pipe = hf_pipeline  # 型別未知，但我們只在 __call__ 中集中處理/檢查

    def __call__(self, prompt: str, /) -> list[GenOutput]:
        raw = self._pipe(prompt)

        # （嚴謹）基本型別驗證與轉換；避免第三方輸出格式變更時靜默錯誤
        if not isinstance(raw, list):
            raise TypeError(f"HF pipeline returned non-list: {type(raw)!r}")

        out: list[GenOutput] = []
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                raise TypeError(f"HF pipeline item[{i}] not dict: {type(item)!r}")
            text = item.get("generated_text")
            if not isinstance(text, str):
                raise TypeError(f"HF pipeline item[{i}]['generated_text'] not str: {type(text)!r}")
            out.append(GenOutput(generated_text=text))
        return out

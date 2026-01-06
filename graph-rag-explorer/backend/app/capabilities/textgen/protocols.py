from __future__ import annotations
from typing import Protocol
from app.capabilities.textgen.types import GenOutput

class TextGenPipe(Protocol):
    """抽象：任何『文本生成器』都要接受字串，回傳標準化的生成結果列表。"""
    def __call__(self, prompt: str, /) -> list[GenOutput]: ...

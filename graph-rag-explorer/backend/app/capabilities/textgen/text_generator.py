from __future__ import annotations

from typing import Protocol, TypedDict, List


class GeneratedText(TypedDict):
    """
    單筆文字生成結果（對齊 transformers pipeline 輸出最小子集）
    """
    generated_text: str


class TextGenerator(Protocol):
    """
    TextGenerator 是「文字生成能力」的最小行為介面。

    任何能根據 prompt 產生文字的物件，都可以實作此介面，
    不論底層是 transformers、OpenAI API、或測試用假物件。
    """

    def generate(self, prompt: str) -> List[GeneratedText]:
        """
        根據 prompt 產生文字。

        Args:
            prompt: 輸入提示詞。

        Returns:
            文字生成結果清單（至少包含 generated_text 欄位）。
        """
        ...

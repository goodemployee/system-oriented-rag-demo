# app/core/llm.py
from __future__ import annotations
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase, pipeline
import torch
import os

from app.capabilities.textgen.protocols import TextGenPipe
from app.capabilities.textgen.text_generator import GeneratedText


class LLM:
    """
    通用文字生成模型。
    可被 GraphExtractor 共用。
    """

    def __init__(self, model_id: str, device: Optional[str] = None) -> None:
        self.model_id: str = model_id
        self.device: str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.model: PreTrainedModel | None = None
        self.pipe: Optional[TextGenPipe] = None

    # -------------------------------------------------------------
    # 模型載入 / 釋放
    # -------------------------------------------------------------
    def load(self) -> None:
        """載入 tokenizer、模型與生成管線"""
        print(f"🦙 載入 LLM 模型：{self.model_id} ({self.device})")

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16, 
            device_map=None,
            low_cpu_mem_usage=True
        )

        self.pipe = pipeline( # type: ignore[call-overload]
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer, # type: ignore
            device=0 if self.device == "cuda" else -1,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
        )

    def unload(self) -> None:
        """釋放 GPU 資源"""
        print("🧹 卸載 LLM 模型資源 ...")
        del self.pipe, self.model, self.tokenizer
        torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # 文本生成接口
    # -------------------------------------------------------------
    def answer(self, question: str, passages: list[str]) -> str:
        """生成回答（RAG 的生成階段）"""
        if not self.pipe:
            raise RuntimeError("LLM 尚未初始化。請先呼叫 load()。")

        context = "\n".join(passages)
        prompt = (
            f"[系統]\n你是知識型助手，根據以下內容回答問題。\n"
            f"[內容]\n{context}\n"
            f"[問題]\n{question}\n"
            f"請給出清晰、簡潔的回答："
        )

        result = self.pipe(prompt)[0]["generated_text"]
        return result.strip()
    
    def generate(self, prompt: str) -> List[GeneratedText]:
        if self.pipe is None:
            raise RuntimeError("LLM 尚未初始化")

        return self.pipe(prompt)

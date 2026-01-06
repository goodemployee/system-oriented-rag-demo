from typing import Optional
import torch

from app.config.modules import ModulesConfig
from app.core.embedding.embedder import Embedder
from app.core.llm.llm import LLM
from app.core.graph.graph_extractor import GraphExtractor

class ModelRegistry:
    """
    統一管理所有模型實例（LLM / Embedder / GraphExtractor）。
    負責載入、共用、釋放與類型安全控制。

    所有路徑由各模型內部透過 app.paths 管理。
    """

    def __init__(
        self,
        modules: ModulesConfig | None = None,
        device: str | None = None,
    ) -> None:
        self.device: str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.modules: ModulesConfig = modules or ModulesConfig()

        # 模型資源
        self._embedder: Optional[Embedder] = None
        self._llm: Optional[LLM] = None

    # === 載入流程 ===
    ### 主動初始化並放入快取
    def load_all(self) -> None:
        print(f"🚀 初始化模型 (device={self.device}) ...")
        self._embedder = self.load_embedder()
        self._llm = self.load_llm()
        print("✅ 所有模型初始化完成！")

    ### 把 embedder 做好並回傳(如果不接會空發)
    def load_embedder(self) -> Embedder:
        """Embedder 通常放 CPU"""
        embedder  = Embedder(
            model_id=self.modules.embedder_model,
            device="cpu",
        )
        print(f"✅ Embedder ready ({self.modules.embedder_model})")
        return embedder

    ### 把 llm 做好並回傳(如果不接會空發)
    def load_llm(self) -> LLM:
        """載入共用 LLM，用於生成答案與圖譜抽取"""
        print("🦙 初始化 LLM ...")
        llm = LLM(
            model_id=self.modules.llm_model,
            device=self.device,
        )
        llm.load()
        print(f"✅ LLM ready ({self.modules.llm_model})")
        return llm

    # === 釋放流程（可選） ===
    def unload_all(self) -> None:
        print("🧹 釋放所有模型資源 ...")

        if self._embedder and hasattr(self._embedder, "unload"):
            self._embedder.unload()

        if self._llm and hasattr(self._llm, "unload"):
            self._llm.unload()

        torch.cuda.empty_cache()
        print("✅ 資源釋放完畢")

    # === 型別安全的 getter ===
    ### 提供Embedder快取
    def _get_embedder_internal(self) -> Embedder:
        if self._embedder is None:
            self._embedder = self.load_embedder()
        
        if self._embedder is None:
            raise RuntimeError("Embedder 尚未載入")
        
        return self._embedder

    ### 提供LLM快取
    def _get_llm_internal(self) -> LLM:
        if self._llm is None:
            self._llm = self.load_llm()
        
        if self._llm is None:
            raise RuntimeError("LLM 尚未載入")
        
        return self._llm

    ### === embedder的封裝 ===
    def add_chunks(self, texts: list[str]) -> None:
        embedder = self._get_embedder_internal()
        embedder.add_chunks(texts)

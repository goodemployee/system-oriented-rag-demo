"""
預先下載並轉存所有模型，使 backend 可離線啟動。
包含：
 - TinyLlama (LLM)
 - all-MiniLM-L6-v2 (Embedder)
 - Phi-3.5-mini-instruct (Graph Extractor)
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def prefetch_llm():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    target = MODELS_DIR / "llm"
    target.mkdir(parents=True, exist_ok=True)

    print(f"🦙 下載 LLM：{model_id}")
    AutoModelForCausalLM.from_pretrained(model_id, cache_dir=target)
    AutoTokenizer.from_pretrained(model_id, cache_dir=target)
    print(f"✅ LLM 已快取至 {target}")


def prefetch_embedder():
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    target = MODELS_DIR / "embedder"
    target.mkdir(parents=True, exist_ok=True)

    print(f"🔤 下載 Embedder：{model_id}")
    # 使用 SentenceTransformers 官方 API 下載並轉存成可直接載入的格式
    model = SentenceTransformer(model_id)
    model.save(str(target))
    print(f"✅ Embedder 已保存為可直接載入格式於 {target}")


def prefetch_graph_extractor():
    model_id = "microsoft/Phi-3.5-mini-instruct"
    target = MODELS_DIR / "graph_extractor"
    target.mkdir(parents=True, exist_ok=True)

    print(f"🧠 下載 Graph Extractor：{model_id}")
    AutoModelForCausalLM.from_pretrained(model_id, cache_dir=target)
    AutoTokenizer.from_pretrained(model_id, cache_dir=target)
    print(f"✅ Graph Extractor 已快取至 {target}")


if __name__ == "__main__":
    print("🚀 開始預下載所有模型 ...")
    prefetch_llm()
    prefetch_embedder()
    prefetch_graph_extractor()
    print("🎉 所有模型已準備完成，可離線啟動！")

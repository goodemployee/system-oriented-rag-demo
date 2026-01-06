# backend/app/paths.py
from __future__ import annotations
import os
from pathlib import Path

# 以 backend/ 為根（因為 launch.json 的 cwd 已經設為 backend）:contentReference[oaicite:3]{index=3}
BACKEND_ROOT = Path(os.getenv("BACKEND_ROOT", Path.cwd()))
DATA_DIR = Path(os.getenv("DATA_DIR", BACKEND_ROOT / "data"))

# --- Storage paths (資料) ---
# 已上傳檔案留存
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", DATA_DIR / "uploads"))

# 向量資料庫
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", DATA_DIR / "chroma" / "database"))

# 三元圖
GRAPH_STORE_PATH = Path(os.getenv("GRAPH_STORE_PATH", DATA_DIR / "graph" / "graph_store.json"))


# --- Model cache paths (預載入模型) ---
MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", DATA_DIR / "models_cache"))

EMBEDDER_CACHE_DIR = Path(os.getenv("EMBEDDER_CACHE_DIR", MODEL_CACHE_DIR / "embedder_cache"))

GRAPH_EXTRACTOR_CACHE_DIR = Path(os.getenv("GRAPH_EXTRACTOR_CACHE_DIR", MODEL_CACHE_DIR / "graph_extractor_cache"))


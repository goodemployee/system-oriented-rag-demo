# app/globals.py
from typing import Optional
from app.infrastructure.models.model_loader import ModelRegistry

_REGISTRY: Optional[ModelRegistry] = None

def set_registry(reg: ModelRegistry) -> None:
    global _REGISTRY
    _REGISTRY = reg

def get_registry() -> ModelRegistry:
    if _REGISTRY is None:
        raise RuntimeError("ModelRegistry 尚未初始化")
    return _REGISTRY

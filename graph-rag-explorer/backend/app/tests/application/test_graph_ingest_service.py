from typing import Any, List

from app.capabilities.textgen.text_generator import GeneratedText
from app.core.graph.graph_extractor import GraphExtractor
from app.core.graph.graph_store import Triple


class _DummyPipe:
    """測試用假 pipe，不會真的被呼叫"""
    def __call__(self, prompt: str) -> List[dict[str, Any]]:
        raise AssertionError("這個測試不應該呼叫 pipe")


class _DummyLLM:
    def generate(self, prompt: str) -> List[GeneratedText]:
        raise AssertionError("這個測試不應該呼叫 generate")


def test_缺少_object_時_normalize_triples_會自動補齊():
    extractor = GraphExtractor(
        llm=_DummyLLM(),  # ✅ 滿足初始化條件
    )

    raw_triples = [
        {"subject": "星爆", "predicate": "攻速"}
    ]

    normalized = extractor._normalize_triples(raw_triples)

    assert normalized == [
        {
            "subject": "星爆",
            "predicate": "隱含",
            "object": "攻速",
        }
    ]

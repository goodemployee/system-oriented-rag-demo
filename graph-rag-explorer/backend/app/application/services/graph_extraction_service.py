from app.core.graph.graph_extractor import GraphExtractor
from app.application.types.graph_triple import GraphTriple
from app.infrastructure.models.model_provider import ModelProvider


class GraphExtractionService:
    def __init__(self, provider: ModelProvider):
        self.provider = provider

    def extract(self, text: str) -> list[GraphTriple]:
        llm = self.provider.get_llm()
        extractor = GraphExtractor(llm=llm)

        raw_triples = extractor.extract_triples(text)

        return [
            GraphTriple(
                subject=t["subject"],
                predicate=t["predicate"],
                object=t["object"],
                confidence=self._parse_confidence(t.get("confidence")),
                source_text=text,
            )
            for t in raw_triples
        ]
    
    def _parse_confidence(self, value) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None


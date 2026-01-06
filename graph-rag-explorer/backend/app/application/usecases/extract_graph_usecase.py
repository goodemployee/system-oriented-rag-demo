from typing import Any

from app.application.services.graph_ingest_service import GraphIngestService


class ExtractGraphUseCase:
    def __init__(self, ingest_service: GraphIngestService):
        self.ingest_service = ingest_service

    def execute(self, file_path: str, max_chunks: int) -> dict[str, Any]:
        triples = self.ingest_service.ingest_from_file(
            file_path=file_path,
            max_chunks=max_chunks,
        )

        return {
            "file_path": str(file_path),
            "triples": triples,
            "count": len(triples),
        }

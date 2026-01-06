from typing import Any

from app.application.services.retrieval_service import RetrievalService
from app.application.services.answer_generation_service import AnswerGenerationService
from app.application.services.graph_extraction_service import GraphExtractionService


class AskQuestionUseCase:
    def __init__(
        self,
        retrieval: RetrievalService,
        answer_generator: AnswerGenerationService,
        graph_extractor: GraphExtractionService,
    ):
        self.retrieval = retrieval
        self.answer_generator = answer_generator
        self.graph_extractor = graph_extractor

    def execute(self, question: str) -> dict[str, Any]:
        passages = self.retrieval.retrieve(question)
        answer = self.answer_generator.generate(question, passages)
        triples = self.graph_extractor.extract(answer)

        return {
            "question": question,
            "answer": answer,
            "triples": triples,
        }

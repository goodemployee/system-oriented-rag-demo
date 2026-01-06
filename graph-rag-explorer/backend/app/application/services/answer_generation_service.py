from app.infrastructure.models.model_provider import ModelProvider


class AnswerGenerationService:
    def __init__(self, provider: ModelProvider):
        self.provider = provider
        
    def generate(self, question: str, passages: list[str]) -> str:
        llm = self.provider.get_llm()
        return llm.answer(question, passages)

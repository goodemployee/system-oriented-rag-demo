from unittest.mock import Mock

from app.application.usecases.ask_question_usecase import AskQuestionUseCase
from app.application.types.graph_triple import GraphTriple


def test_ask_question_usecase_happy_path():
    # Arrange
    question = "What is GraphRAG?"

    mock_retrieval = Mock()
    mock_answer_generator = Mock()
    mock_graph_extractor = Mock()

    mock_retrieval.retrieve.return_value = [
        "GraphRAG combines graphs with retrieval."
    ]

    mock_answer_generator.generate.return_value = (
        "GraphRAG is a method that combines knowledge graphs with RAG."
    )

    mock_graph_extractor.extract.return_value = [
        GraphTriple(
            subject="GraphRAG",
            predicate="combines",
            object="knowledge graphs",
            confidence=0.9,
        )
    ]

    usecase = AskQuestionUseCase(
        retrieval=mock_retrieval,
        answer_generator=mock_answer_generator,
        graph_extractor=mock_graph_extractor,
    )

    # Act
    result = usecase.execute(question)

    # Assert：回傳結果
    assert result["question"] == question
    assert result["answer"].startswith("GraphRAG is")
    assert len(result["triples"]) == 1
    assert result["triples"][0].predicate == "combines"

    # Assert：互動驗證（非常重要）
    mock_retrieval.retrieve.assert_called_once_with(question)
    mock_answer_generator.generate.assert_called_once_with(
        question, mock_retrieval.retrieve.return_value
    )
    mock_graph_extractor.extract.assert_called_once_with(
        mock_answer_generator.generate.return_value
    )

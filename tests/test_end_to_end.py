import pytest
from evaluation.pipeline_evaluator import PipelineEvaluator


@pytest.fixture
def evaluator():
    return PipelineEvaluator()


def test_full_pipeline_pass(evaluator):
    retrieved = [
        "RAG stands for Retrieval-Augmented Generation.",
        "It combines retrieval with language model generation.",
    ]
    result = evaluator.evaluate(
        retrieved=retrieved,
        relevant_ids=[0, 1],
        expected_facts=["RAG uses retrieval", "RAG uses generation"],
        answer_claims=["RAG combines retrieval and generation."],
        ground_truth_facts=["RAG combines retrieval and generation."],
    )
    assert result.context_precision >= 0.70
    assert result.context_recall >= 0.75
    assert result.hallucination_rate <= 0.15
    assert result.answer_correctness >= 0.80


def test_pipeline_flags_hallucination(evaluator):
    retrieved = ["RAG uses vector search for retrieval."]
    result = evaluator.evaluate(
        retrieved=retrieved,
        relevant_ids=[0],
        expected_facts=["RAG uses vector search"],
        answer_claims=["RAG uses quantum computing for retrieval."],
        ground_truth_facts=["RAG uses vector search"],
    )
    assert result.hallucination_rate > 0.0

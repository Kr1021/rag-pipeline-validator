import pytest
from validators.faithfulness_scorer import FaithfulnessScorer


@pytest.fixture
def scorer():
    return FaithfulnessScorer(threshold=0.85)


def test_fully_grounded_answer(scorer):
    context = ["The Eiffel Tower is located in Paris, France."]
    answer_claims = ["The Eiffel Tower is in Paris."]
    result = scorer.score(answer_claims, context)
    assert result.score >= 0.85


def test_hallucinated_answer(scorer):
    context = ["The Eiffel Tower is located in Paris, France."]
    answer_claims = ["The Eiffel Tower is in Berlin."]
    result = scorer.score(answer_claims, context)
    assert result.score < 0.85


def test_empty_claims(scorer):
    result = scorer.score([], ["Some context."])
    assert result.score == 1.0


def test_threshold_pass(scorer):
    context = ["Python is a high-level programming language."]
    answer_claims = ["Python is a programming language."]
    result = scorer.score(answer_claims, context)
    assert scorer.passes(result)

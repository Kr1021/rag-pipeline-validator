from dataclasses import dataclass
from typing import List


@dataclass
class CorrectnessResult:
    score: float
    matched: int
    total: int


class AnswerCorrectnessValidator:
    def __init__(self, threshold: float = 0.80):
        self.threshold = threshold

    def score(self, predicted_facts: List[str], ground_truth_facts: List[str]) -> CorrectnessResult:
        if not ground_truth_facts:
            return CorrectnessResult(score=1.0, matched=0, total=0)

        matched = sum(
            1 for fact in ground_truth_facts
            if any(fact.lower() in pred.lower() for pred in predicted_facts)
        )
        correctness = matched / len(ground_truth_facts)

        return CorrectnessResult(
            score=round(correctness, 4),
            matched=matched,
            total=len(ground_truth_facts),
        )

    def passes(self, result: CorrectnessResult) -> bool:
        return result.score >= self.threshold

from dataclasses import dataclass
from typing import List


@dataclass
class RecallResult:
    score: float
    covered: int
    total_expected: int


class ContextRecallValidator:
    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold

    def score(self, retrieved_chunks: List[str], expected_facts: List[str]) -> RecallResult:
        if not expected_facts:
            return RecallResult(score=1.0, covered=0, total_expected=0)

        covered = sum(
            1 for fact in expected_facts
            if any(fact.lower() in chunk.lower() for chunk in retrieved_chunks)
        )
        recall = covered / len(expected_facts)

        return RecallResult(
            score=round(recall, 4),
            covered=covered,
            total_expected=len(expected_facts),
        )

    def passes(self, result: RecallResult) -> bool:
        return result.score >= self.threshold

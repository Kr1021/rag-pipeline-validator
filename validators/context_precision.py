from dataclasses import dataclass
from typing import List


@dataclass
class PrecisionResult:
    score: float
    relevant_chunks: int
    total_chunks: int


class ContextPrecisionValidator:
    def __init__(self, threshold: float = 0.70):
        self.threshold = threshold

    def score(self, retrieved_chunks: List[str], relevant_chunk_ids: List[int]) -> PrecisionResult:
        if not retrieved_chunks:
            return PrecisionResult(score=0.0, relevant_chunks=0, total_chunks=0)

        relevant = sum(1 for i, _ in enumerate(retrieved_chunks) if i in relevant_chunk_ids)
        precision = relevant / len(retrieved_chunks)

        return PrecisionResult(
            score=round(precision, 4),
            relevant_chunks=relevant,
            total_chunks=len(retrieved_chunks),
        )

    def passes(self, result: PrecisionResult) -> bool:
        return result.score >= self.threshold

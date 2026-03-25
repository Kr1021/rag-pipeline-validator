from dataclasses import dataclass
from typing import List


@dataclass
class HallucinationResult:
    hallucination_rate: float
    grounded_claims: int
    total_claims: int
    ungrounded: List[str]


class HallucinationDetector:
    def __init__(self, threshold: float = 0.15):
        self.threshold = threshold

    def detect(self, answer_claims: List[str], context_chunks: List[str]) -> HallucinationResult:
        ungrounded = [
            claim for claim in answer_claims
            if not any(claim.lower() in chunk.lower() for chunk in context_chunks)
        ]
        rate = len(ungrounded) / len(answer_claims) if answer_claims else 0.0

        return HallucinationResult(
            hallucination_rate=round(rate, 4),
            grounded_claims=len(answer_claims) - len(ungrounded),
            total_claims=len(answer_claims),
            ungrounded=ungrounded,
        )

    def passes(self, result: HallucinationResult) -> bool:
        return result.hallucination_rate <= self.threshold

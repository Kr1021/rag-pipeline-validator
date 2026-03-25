from dataclasses import dataclass
from typing import List

from validators.context_precision import ContextPrecisionValidator
from validators.context_recall import ContextRecallValidator
from validators.hallucination_detector import HallucinationDetector
from validators.answer_correctness import AnswerCorrectnessValidator


@dataclass
class PipelineEvalResult:
    context_precision: float
    context_recall: float
    hallucination_rate: float
    answer_correctness: float
    passed: bool


class PipelineEvaluator:
    def __init__(self):
        self.precision = ContextPrecisionValidator()
        self.recall = ContextRecallValidator()
        self.hallucination = HallucinationDetector()
        self.correctness = AnswerCorrectnessValidator()

    def evaluate(self, retrieved: List[str], relevant_ids: List[int],
                 expected_facts: List[str], answer_claims: List[str],
                 ground_truth_facts: List[str]) -> PipelineEvalResult:

        p = self.precision.score(retrieved, relevant_ids)
        r = self.recall.score(retrieved, expected_facts)
        h = self.hallucination.detect(answer_claims, retrieved)
        c = self.correctness.score(answer_claims, ground_truth_facts)

        passed = all([
            self.precision.passes(p),
            self.recall.passes(r),
            self.hallucination.passes(h),
            self.correctness.passes(c),
        ])

        return PipelineEvalResult(
            context_precision=p.score,
            context_recall=r.score,
            hallucination_rate=h.hallucination_rate,
            answer_correctness=c.score,
            passed=passed,
        )

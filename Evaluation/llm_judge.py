from dataclasses import dataclass
from typing import Optional


@dataclass
class JudgementResult:
    score: float
    reasoning: str
    passed: bool


class LLMJudge:
    def __init__(self, model: str = "gpt-4o", threshold: float = 0.80):
        self.model = model
        self.threshold = threshold

    def build_prompt(self, question: str, answer: str, context: str, ground_truth: Optional[str] = None) -> str:
        base = (
            f"Question: {question}\n"
            f"Context: {context}\n"
            f"Answer: {answer}\n"
        )
        if ground_truth:
            base += f"Ground Truth: {ground_truth}\n"
        base += "\nRate the answer quality from 0.0 to 1.0 and explain your reasoning."
        return base

    def judge(self, question: str, answer: str, context: str, ground_truth: Optional[str] = None) -> JudgementResult:
        prompt = self.build_prompt(question, answer, context, ground_truth)
        # Placeholder: replace with actual LLM call
        score = 0.85
        reasoning = "Answer is well-grounded in the provided context."
        return JudgementResult(score=score, reasoning=reasoning, passed=score >= self.threshold)

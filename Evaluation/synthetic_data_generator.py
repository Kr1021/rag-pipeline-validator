from dataclasses import dataclass
from typing import List


@dataclass
class SyntheticSample:
    question: str
    context: str
    ground_truth: str
    question_type: str


class SyntheticDataGenerator:
    QUESTION_TYPES = ["factoid", "multi-hop", "comparison", "summarisation"]

    def __init__(self, model: str = "gpt-4o"):
        self.model = model

    def generate(self, source_text: str, n_samples: int = 5) -> List[SyntheticSample]:
        # Placeholder: replace with actual LLM-driven generation
        samples = []
        for i in range(n_samples):
            samples.append(SyntheticSample(
                question=f"Sample question {i+1} from source?",
                context=source_text[:200],
                ground_truth=f"Expected answer {i+1}",
                question_type=self.QUESTION_TYPES[i % len(self.QUESTION_TYPES)],
            ))
        return samples

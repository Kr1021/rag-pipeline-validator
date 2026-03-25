from dataclasses import dataclass
from typing import List, Dict


@dataclass
class AgentTrace:
    steps: List[Dict]
    final_answer: str
    total_iterations: int


@dataclass
class AgenticEvalResult:
    planning_score: float
    tool_use_score: float
    iteration_efficiency: float
    overall: float


class AgenticEvaluator:
    def __init__(self, max_iterations: int = 10):
        self.max_iterations = max_iterations

    def evaluate(self, trace: AgentTrace) -> AgenticEvalResult:
        planning = self._score_planning(trace.steps)
        tool_use = self._score_tool_use(trace.steps)
        efficiency = 1.0 - (trace.total_iterations / self.max_iterations)

        overall = round((planning + tool_use + max(efficiency, 0)) / 3, 4)
        return AgenticEvalResult(
            planning_score=planning,
            tool_use_score=tool_use,
            iteration_efficiency=round(max(efficiency, 0), 4),
            overall=overall,
        )

    def _score_planning(self, steps: List[Dict]) -> float:
        planned = sum(1 for s in steps if s.get("type") == "plan")
        return round(min(planned / 3, 1.0), 4)

    def _score_tool_use(self, steps: List[Dict]) -> float:
        tool_calls = sum(1 for s in steps if s.get("type") == "tool_call")
        return round(min(tool_calls / 5, 1.0), 4)

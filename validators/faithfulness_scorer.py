from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from sentence_transformers import SentenceTransformer, util


@dataclass
class FaithfulnessResult:
      query: str
      answer: str
      context: str
      faithfulness_score: float
      answer_relevance: float
      unsupported_statements: List[str] = field(default_factory=list)
      passed: bool = True


class FaithfulnessScorer:
      def __init__(
                self,
                model_name: str = "gpt-4",
                embedding_model: str = "all-MiniLM-L6-v2",
                faithfulness_threshold: float = 0.75,
                relevance_threshold: float = 0.65,
      ):
                self.llm = ChatOpenAI(model_name=model_name, temperature=0.0)
                self.embedder = SentenceTransformer(embedding_model)
                self.faithfulness_threshold = faithfulness_threshold
                self.relevance_threshold = relevance_threshold

      def _decompose_answer(self, answer: str) -> List[str]:
                prompt = (
                              "Break the following answer into individual atomic statements. "
                              "Return each statement on a new line.\n\nAnswer:\n" + answer
                )
                result = self.llm([HumanMessage(content=prompt)])
                return [s.strip() for s in result.content.strip().split("\n") if s.strip()]

      def _statement_supported(self, statement: str, context: str) -> float:
                prompt = (
                              f"Is the following statement directly supported by the context? "
                              f"Return a float between 0.0 (not supported) and 1.0 (fully supported).\n\n"
                              f"Context:\n{context}\n\nStatement:\n{statement}\n\nScore:"
                )
                result = self.llm([HumanMessage(content=prompt)])
                try:
                              return float(result.content.strip())
except ValueError:
            return 0.0

    def _answer_relevance(self, query: str, answer: str) -> float:
              embeddings = self.embedder.encode([query, answer], convert_to_tensor=True)
              return float(util.cos_sim(embeddings[0], embeddings[1]))

    def score(self, query: str, answer: str, context: str) -> FaithfulnessResult:
              statements = self._decompose_answer(answer)
              if not statements:
                            return FaithfulnessResult(
                                              query=query,
                                              answer=answer,
                                              context=context,
                                              faithfulness_score=0.0,
                                              answer_relevance=0.0,
                                              passed=False,
                            )

              support_scores = [self._statement_supported(s, context) for s in statements]
              faithfulness = float(np.mean(support_scores))
              unsupported = [
                  statements[i]
                  for i, score in enumerate(support_scores)
                  if score < self.faithfulness_threshold
              ]
              relevance = self._answer_relevance(query, answer)

        return FaithfulnessResult(
                      query=query,
                      answer=answer,
                      context=context,
                      faithfulness_score=faithfulness,
                      answer_relevance=relevance,
                      unsupported_statements=unsupported,
                      passed=(
                                        faithfulness >= self.faithfulness_threshold
                                        and relevance >= self.relevance_threshold
                      ),
        )

    def batch_score(self, records: List[dict]) -> List[FaithfulnessResult]:
              return [
                            self.score(r["query"], r["answer"], r["context"])
                            for r in records
              ]

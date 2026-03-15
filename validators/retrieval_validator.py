from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer, util


@dataclass
class RetrievalValidationResult:
      query: str
      retrieved_chunks: List[str]
      context_recall: float
      context_precision: float
      mrr: float
      passed: bool
      metadata: Dict[str, Any] = field(default_factory=dict)


class RetrievalValidator:
      def __init__(
                self,
                embedding_model: str = "all-MiniLM-L6-v2",
                recall_threshold: float = 0.6,
                precision_threshold: float = 0.5,
      ):
                self.embedder = SentenceTransformer(embedding_model)
                self.recall_threshold = recall_threshold
                self.precision_threshold = precision_threshold

      def _embed(self, texts: List[str]):
                return self.embedder.encode(texts, convert_to_tensor=True)

      def _context_recall(
                self, query_embedding, chunk_embeddings, top_k: int
      ) -> float:
                sims = [
                              float(util.cos_sim(query_embedding, ce))
                              for ce in chunk_embeddings
                ]
                top_sims = sorted(sims, reverse=True)[:top_k]
                return float(np.mean(top_sims)) if top_sims else 0.0

      def _context_precision(
                self, query_embedding, chunk_embeddings, threshold: float = 0.5
      ) -> float:
                sims = [
                              float(util.cos_sim(query_embedding, ce))
                              for ce in chunk_embeddings
                ]
                relevant = [s for s in sims if s >= threshold]
                return len(relevant) / len(sims) if sims else 0.0

      def _mean_reciprocal_rank(
                self, query_embedding, chunk_embeddings, relevance_threshold: float = 0.6
      ) -> float:
                sims = [
                              float(util.cos_sim(query_embedding, ce))
                              for ce in chunk_embeddings
                ]
                ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
                for rank, (_, score) in enumerate(ranked, start=1):
                              if score >= relevance_threshold:
                                                return 1.0 / rank
                                        return 0.0

    def validate(
              self,
              query: str,
              retrieved_chunks: List[str],
              top_k: int = 5,
    ) -> RetrievalValidationResult:
              query_emb = self._embed([query])[0]
              chunk_embs = self._embed(retrieved_chunks)

        recall = self._context_recall(query_emb, chunk_embs, top_k)
        precision = self._context_precision(query_emb, chunk_embs)
        mrr = self._mean_reciprocal_rank(query_emb, chunk_embs)

        passed = recall >= self.recall_threshold and precision >= self.precision_threshold

        return RetrievalValidationResult(
                      query=query,
                      retrieved_chunks=retrieved_chunks,
                      context_recall=recall,
                      context_precision=precision,
                      mrr=mrr,
                      passed=passed,
                      metadata={"top_k": top_k, "chunk_count": len(retrieved_chunks)},
        )

    def batch_validate(
              self, records: List[Dict[str, Any]], top_k: int = 5
    ) -> List[RetrievalValidationResult]:
              return [
                            self.validate(r["query"], r["retrieved_chunks"], top_k)
                            for r in records
              ]

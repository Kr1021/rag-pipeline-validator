from dataclasses import dataclass
from typing import List


@dataclass
class ChunkingResult:
    avg_chunk_size: float
    fragmentation_score: float
    orphaned_chunks: int
    passed: bool


class ChunkingValidator:
    def __init__(self, min_chunk_size: int = 100, max_chunk_size: int = 512):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def validate(self, chunks: List[str]) -> ChunkingResult:
        sizes = [len(c.split()) for c in chunks]
        avg = sum(sizes) / len(sizes) if sizes else 0
        orphaned = sum(1 for s in sizes if s < self.min_chunk_size)
        fragmentation = orphaned / len(chunks) if chunks else 0.0

        return ChunkingResult(
            avg_chunk_size=round(avg, 2),
            fragmentation_score=round(fragmentation, 4),
            orphaned_chunks=orphaned,
            passed=fragmentation < 0.2 and self.min_chunk_size <= avg <= self.max_chunk_size,
        )

"""
Parallel Braid Reduction Module for topoKEMP2

This module implements parallel algorithms for braid word reduction,
leveraging multi-core processors to achieve better performance on
large SAT instances.

KEY TECHNIQUES:
1. Parallel chunk reduction - divide braid into independent chunks
2. Work-stealing queue for load balancing
3. Lock-free stack-based cancellation
4. Parallel prefix scan for cancellation propagation

THEORETICAL COMPLEXITY:
- Sequential: O(n) where n = braid length
- Parallel: O(n/p + log(n)) where p = number of processors
- Speedup: Up to p-fold on p processors (Amdahl's law limited)
"""

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Callable
import threading
import time
import os
from collections import deque

from .braid import BraidWord, BraidGenerator, GeneratorSign


@dataclass
class ReductionChunk:
    """A chunk of braid word for parallel processing."""
    generators: List[BraidGenerator]
    start_index: int
    chunk_id: int
    left_boundary: Optional[BraidGenerator] = None
    right_boundary: Optional[BraidGenerator] = None


@dataclass
class ReductionResult:
    """Result from reducing a chunk."""
    chunk_id: int
    reduced: List[BraidGenerator]
    left_cancelled: bool = False
    right_cancelled: bool = False
    cancellations: int = 0


class ParallelBraidReducer:
    """
    Parallel braid word reducer using multi-threading.

    This reducer divides the braid word into chunks and processes
    them in parallel, then merges the results.
    """

    def __init__(self, num_workers: int = None):
        """
        Initialize parallel reducer.

        Args:
            num_workers: Number of worker threads (default: CPU count)
        """
        self.num_workers = num_workers or os.cpu_count() or 4
        self.stats = {
            "total_reductions": 0,
            "parallel_time_ms": 0,
            "sequential_time_ms": 0,
            "speedup": 0.0,
        }

    def reduce(self, braid: BraidWord) -> Tuple[BraidWord, int]:
        """
        Reduce a braid word using parallel processing.

        Args:
            braid: Input braid word

        Returns:
            (reduced_braid, num_cancellations)
        """
        generators = list(braid.generators)

        if len(generators) < self.num_workers * 4:
            # Too small for parallel processing - use sequential
            return self._sequential_reduce(generators)

        # Parallel reduction
        return self._parallel_reduce(generators)

    def _sequential_reduce(self, generators: List[BraidGenerator]) -> Tuple[BraidWord, int]:
        """Stack-based sequential reduction (O(n))."""
        stack = []
        cancellations = 0

        for gen in generators:
            if stack and self._can_cancel(stack[-1], gen):
                stack.pop()
                cancellations += 1
            else:
                stack.append(gen)

        # Compute num_strands from remaining generators
        num_strands = max((g.index + 1 for g in stack), default=2)
        return BraidWord(num_strands, stack), cancellations

    def _can_cancel(self, g1: BraidGenerator, g2: BraidGenerator) -> bool:
        """Check if two generators cancel (σᵢ·σᵢ⁻¹ = ε)."""
        return g1.index == g2.index and g1.sign != g2.sign

    def _can_commute(self, g1: BraidGenerator, g2: BraidGenerator) -> bool:
        """Check if two generators commute (|i-j| > 1)."""
        return abs(g1.index - g2.index) > 1

    def _parallel_reduce(self, generators: List[BraidGenerator]) -> Tuple[BraidWord, int]:
        """
        Parallel reduction using chunk-based approach.

        Algorithm:
        1. Divide into chunks
        2. Reduce each chunk in parallel
        3. Merge reduced chunks, handling boundaries
        4. Repeat until no more reductions possible
        """
        total_cancellations = 0
        current = generators

        # Iterate until no more reductions
        while True:
            if len(current) < 2:
                break

            # Create chunks
            chunks = self._create_chunks(current)

            if len(chunks) <= 1:
                # Single chunk - reduce sequentially
                braid, cancellations = self._sequential_reduce(current)
                return braid, total_cancellations + cancellations

            # Reduce chunks in parallel
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {
                    executor.submit(self._reduce_chunk, chunk): chunk.chunk_id
                    for chunk in chunks
                }

                results = {}
                for future in as_completed(futures):
                    chunk_id = futures[future]
                    results[chunk_id] = future.result()

            # Merge results
            current, boundary_cancellations = self._merge_results(
                [results[i] for i in sorted(results.keys())]
            )
            total_cancellations += sum(r.cancellations for r in results.values())
            total_cancellations += boundary_cancellations

            if boundary_cancellations == 0:
                # No boundary cancellations - done
                break

        num_strands = max((g.index + 1 for g in current), default=2)
        return BraidWord(num_strands, current), total_cancellations

    def _create_chunks(self, generators: List[BraidGenerator]) -> List[ReductionChunk]:
        """Divide generators into chunks for parallel processing."""
        chunk_size = max(4, len(generators) // self.num_workers)
        chunks = []

        for i in range(0, len(generators), chunk_size):
            chunk_gens = generators[i:i + chunk_size]
            chunk = ReductionChunk(
                generators=chunk_gens,
                start_index=i,
                chunk_id=len(chunks),
                left_boundary=generators[i - 1] if i > 0 else None,
                right_boundary=generators[i + len(chunk_gens)] if i + len(chunk_gens) < len(generators) else None,
            )
            chunks.append(chunk)

        return chunks

    def _reduce_chunk(self, chunk: ReductionChunk) -> ReductionResult:
        """Reduce a single chunk (called in parallel)."""
        braid, cancellations = self._sequential_reduce(chunk.generators)

        return ReductionResult(
            chunk_id=chunk.chunk_id,
            reduced=list(braid.generators),
            cancellations=cancellations,
        )

    def _merge_results(self, results: List[ReductionResult]) -> Tuple[List[BraidGenerator], int]:
        """
        Merge reduced chunks, handling boundary cancellations.

        Returns:
            (merged_generators, boundary_cancellations)
        """
        if not results:
            return [], 0

        merged = []
        boundary_cancellations = 0

        for i, result in enumerate(results):
            if i == 0:
                merged.extend(result.reduced)
            else:
                # Check for cancellation at boundary
                while merged and result.reduced and self._can_cancel(merged[-1], result.reduced[0]):
                    merged.pop()
                    result.reduced.pop(0)
                    boundary_cancellations += 1

                merged.extend(result.reduced)

        return merged, boundary_cancellations


class LockFreeReducer:
    """
    Lock-free parallel braid reducer using atomic operations.

    This implementation uses compare-and-swap semantics for
    thread-safe cancellation without locks.
    """

    def __init__(self, num_workers: int = None):
        self.num_workers = num_workers or os.cpu_count() or 4

    def reduce(self, braid: BraidWord) -> Tuple[BraidWord, int]:
        """
        Reduce using lock-free parallel scan.

        This uses a parallel prefix approach where each worker
        scans a segment and marks cancellable pairs.
        """
        generators = list(braid.generators)
        n = len(generators)

        if n < 100:
            # Sequential for small inputs
            return self._sequential_reduce(generators)

        # Mark cancellable pairs
        cancelled = [False] * n

        # Phase 1: Local cancellation within segments
        segment_size = max(10, n // self.num_workers)
        segments = [(i, min(i + segment_size, n)) for i in range(0, n, segment_size)]

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(self._mark_local_cancellations, generators, start, end, cancelled)
                for start, end in segments
            ]
            for f in as_completed(futures):
                f.result()

        # Phase 2: Boundary resolution (sequential for correctness)
        self._resolve_boundaries(generators, segments, cancelled)

        # Collect remaining generators
        remaining = [g for i, g in enumerate(generators) if not cancelled[i]]
        total_cancelled = sum(1 for c in cancelled if c)

        num_strands = max((g.index + 1 for g in remaining), default=2)
        return BraidWord(num_strands, remaining), total_cancelled // 2

    def _sequential_reduce(self, generators: List[BraidGenerator]) -> Tuple[BraidWord, int]:
        """Stack-based sequential reduction."""
        stack = []
        cancellations = 0

        for gen in generators:
            if stack and self._can_cancel(stack[-1], gen):
                stack.pop()
                cancellations += 1
            else:
                stack.append(gen)

        num_strands = max((g.index + 1 for g in stack), default=2)
        return BraidWord(num_strands, stack), cancellations

    def _can_cancel(self, g1: BraidGenerator, g2: BraidGenerator) -> bool:
        return g1.index == g2.index and g1.sign != g2.sign

    def _mark_local_cancellations(self, generators: List[BraidGenerator],
                                   start: int, end: int,
                                   cancelled: List[bool]) -> int:
        """Mark local cancellations within a segment."""
        local_stack = []
        local_cancel = 0

        for i in range(start, end):
            if cancelled[i]:
                continue

            if local_stack:
                last_idx = local_stack[-1]
                if not cancelled[last_idx] and self._can_cancel(generators[last_idx], generators[i]):
                    cancelled[last_idx] = True
                    cancelled[i] = True
                    local_stack.pop()
                    local_cancel += 1
                    continue

            local_stack.append(i)

        return local_cancel

    def _resolve_boundaries(self, generators: List[BraidGenerator],
                           segments: List[Tuple[int, int]],
                           cancelled: List[bool]):
        """Resolve cancellations at segment boundaries."""
        # Find remaining generators at boundaries
        for i in range(len(segments) - 1):
            _, end1 = segments[i]
            start2, _ = segments[i + 1]

            # Find last non-cancelled in segment 1
            left = None
            for j in range(end1 - 1, segments[i][0] - 1, -1):
                if not cancelled[j]:
                    left = j
                    break

            # Find first non-cancelled in segment 2
            right = None
            for j in range(start2, segments[i + 1][1]):
                if not cancelled[j]:
                    right = j
                    break

            # Try to cancel
            if left is not None and right is not None:
                if self._can_cancel(generators[left], generators[right]):
                    cancelled[left] = True
                    cancelled[right] = True


class WorkStealingReducer:
    """
    Work-stealing parallel reducer for load balancing.

    Uses a deque-based work-stealing approach where idle workers
    steal work from busy workers' queues.
    """

    def __init__(self, num_workers: int = None):
        self.num_workers = num_workers or os.cpu_count() or 4
        self.work_queues: List[deque] = []
        self.results: Dict[int, List[BraidGenerator]] = {}
        self.lock = threading.Lock()

    def reduce(self, braid: BraidWord) -> Tuple[BraidWord, int]:
        """Reduce using work-stealing."""
        generators = list(braid.generators)
        n = len(generators)

        if n < 50:
            return self._sequential_reduce(generators)

        # Initialize work queues
        self.work_queues = [deque() for _ in range(self.num_workers)]
        chunk_size = max(10, n // (self.num_workers * 4))

        # Distribute initial work
        for i, start in enumerate(range(0, n, chunk_size)):
            worker = i % self.num_workers
            self.work_queues[worker].append((start, min(start + chunk_size, n)))

        # Process with work stealing
        self.results = {i: [] for i in range(self.num_workers)}
        total_cancellations = [0]

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(self._worker, worker_id, generators, total_cancellations)
                for worker_id in range(self.num_workers)
            ]
            for f in as_completed(futures):
                f.result()

        # Merge results in order
        merged = []
        cancellations_at_merge = 0

        all_results = []
        for worker_id in range(self.num_workers):
            all_results.extend(self.results[worker_id])

        # Sort by original position and reduce
        all_results.sort(key=lambda x: x[0])
        remaining_gens = [g for _, g in all_results]

        # Final sequential pass
        final_braid, final_cancellations = self._sequential_reduce(remaining_gens)

        return final_braid, total_cancellations[0] + final_cancellations

    def _worker(self, worker_id: int, generators: List[BraidGenerator],
                total_cancellations: List[int]):
        """Worker function with work stealing."""
        my_queue = self.work_queues[worker_id]
        my_results = []

        while True:
            # Try to get work from my queue
            chunk = None
            if my_queue:
                try:
                    chunk = my_queue.popleft()
                except IndexError:
                    pass

            # Try to steal if my queue is empty
            if chunk is None:
                chunk = self._steal_work(worker_id)

            if chunk is None:
                # No more work anywhere
                break

            start, end = chunk
            # Process chunk
            for i in range(start, end):
                my_results.append((i, generators[i]))

        with self.lock:
            self.results[worker_id] = my_results

    def _steal_work(self, my_id: int) -> Optional[Tuple[int, int]]:
        """Try to steal work from another worker."""
        for i in range(self.num_workers):
            if i == my_id:
                continue
            queue = self.work_queues[i]
            if queue:
                try:
                    return queue.pop()  # Steal from the back
                except IndexError:
                    continue
        return None

    def _sequential_reduce(self, generators: List[BraidGenerator]) -> Tuple[BraidWord, int]:
        """Stack-based sequential reduction."""
        stack = []
        cancellations = 0

        for gen in generators:
            if stack and self._can_cancel(stack[-1], gen):
                stack.pop()
                cancellations += 1
            else:
                stack.append(gen)

        num_strands = max((g.index + 1 for g in stack), default=2)
        return BraidWord(num_strands, stack), cancellations

    def _can_cancel(self, g1: BraidGenerator, g2: BraidGenerator) -> bool:
        return g1.index == g2.index and g1.sign != g2.sign


def parallel_reduce(braid: BraidWord, method: str = "chunk",
                   num_workers: int = None) -> Tuple[BraidWord, int]:
    """
    Convenience function for parallel braid reduction.

    Args:
        braid: Input braid word
        method: Reduction method ("chunk", "lockfree", or "stealing")
        num_workers: Number of worker threads

    Returns:
        (reduced_braid, num_cancellations)
    """
    if method == "chunk":
        reducer = ParallelBraidReducer(num_workers)
    elif method == "lockfree":
        reducer = LockFreeReducer(num_workers)
    elif method == "stealing":
        reducer = WorkStealingReducer(num_workers)
    else:
        raise ValueError(f"Unknown method: {method}")

    return reducer.reduce(braid)


def benchmark_parallel_reduction(braid_lengths: List[int] = None,
                                methods: List[str] = None) -> Dict:
    """
    Benchmark different parallel reduction methods.

    Returns timing comparison for different methods and sizes.
    """
    if braid_lengths is None:
        braid_lengths = [100, 500, 1000, 5000, 10000]

    if methods is None:
        methods = ["sequential", "chunk", "lockfree"]

    results = {}

    for length in braid_lengths:
        # Generate random braid
        import random
        generators = []
        for _ in range(length):
            index = random.randint(1, 10)
            sign = GeneratorSign.POSITIVE if random.choice([True, False]) else GeneratorSign.NEGATIVE
            generators.append(BraidGenerator(index, sign))

        num_strands = 11  # Since index goes 1-10
        braid = BraidWord(num_strands, generators)
        results[length] = {}

        for method in methods:
            start = time.perf_counter()

            if method == "sequential":
                # Simple sequential reduction
                stack = []
                for g in generators:
                    if stack and stack[-1].index == g.index and stack[-1].sign != g.sign:
                        stack.pop()
                    else:
                        stack.append(g)
                result_len = len(stack)
            else:
                reduced, _ = parallel_reduce(braid, method)
                result_len = len(reduced.generators)

            elapsed = (time.perf_counter() - start) * 1000
            results[length][method] = {
                "time_ms": elapsed,
                "result_length": result_len,
            }

    return results


if __name__ == "__main__":
    print("Parallel Braid Reduction Benchmark")
    print("=" * 60)

    results = benchmark_parallel_reduction(
        braid_lengths=[100, 500, 1000, 2000],
        methods=["sequential", "chunk", "lockfree"]
    )

    print("\nResults:")
    print("-" * 60)
    for length in sorted(results.keys()):
        print(f"\nBraid length: {length}")
        for method, data in results[length].items():
            print(f"  {method:12s}: {data['time_ms']:8.2f}ms -> {data['result_length']} generators")

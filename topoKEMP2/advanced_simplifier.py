"""
Advanced Braid Simplification Module for topoKEMP2

This module implements advanced braid simplification techniques beyond
simple cancellation, including:

1. Handle Slides (Far Commutativity): σᵢσⱼ = σⱼσᵢ when |i-j| > 1
   - Allows reordering generators to enable more cancellations

2. Braid Relations: σᵢσᵢ₊₁σᵢ = σᵢ₊₁σᵢσᵢ₊₁
   - Can transform braids to enable further simplification

3. Destabilization: Removing σₙ when it doesn't interact
   - Reduces the braid index

THEORETICAL BASIS:
The braid group Bₙ is a non-abelian group with generators {σ₁, ..., σₙ₋₁}
and relations:
  - σᵢσⱼ = σⱼσᵢ for |i-j| > 1
  - σᵢσᵢ₊₁σᵢ = σᵢ₊₁σᵢσᵢ₊₁

Two braids represent the same link if and only if they are related by
a sequence of Markov moves and braid relations.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Set, Dict
from enum import Enum, auto

from .braid import BraidWord, BraidGenerator, GeneratorSign


class SimplificationMove(Enum):
    """Types of simplification moves."""
    CANCELLATION = auto()      # σᵢσᵢ⁻¹ → ε
    HANDLE_SLIDE = auto()      # σᵢσⱼ → σⱼσᵢ (|i-j| > 1)
    BRAID_RELATION = auto()    # σᵢσᵢ₊₁σᵢ → σᵢ₊₁σᵢσᵢ₊₁
    DESTABILIZATION = auto()   # Remove unused strands


@dataclass
class SimplificationStep:
    """Record of a single simplification step."""
    move_type: SimplificationMove
    position: int
    description: str
    before_length: int
    after_length: int


class AdvancedBraidSimplifier:
    """
    Advanced braid word simplifier using multiple techniques.

    This simplifier goes beyond simple free cancellation to use
    the full braid group relations for more aggressive simplification.
    """

    def __init__(self, max_iterations: int = 100, verbose: bool = False):
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.steps: List[SimplificationStep] = []
        self.stats = {
            "cancellations": 0,
            "handle_slides": 0,
            "braid_relations": 0,
            "destabilizations": 0,
        }

    def simplify(self, braid: BraidWord) -> Tuple[BraidWord, List[SimplificationStep]]:
        """
        Simplify a braid word using all available techniques.

        Returns:
            (simplified_braid, list_of_steps)
        """
        self.steps = []
        self.stats = {k: 0 for k in self.stats}

        current = braid
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            initial_length = len(current.generators)

            # Phase 1: Aggressive cancellation with handle slides
            current = self._cancel_with_slides(current)

            # Phase 2: Try braid relations to enable more cancellations
            current = self._apply_braid_relations(current)

            # Phase 3: Destabilization
            current = self._destabilize(current)

            # Check if we made progress
            if len(current.generators) >= initial_length:
                break

        return current, self.steps

    def _cancel_with_slides(self, braid: BraidWord) -> BraidWord:
        """
        Perform cancellation while using handle slides to enable more.

        This uses a greedy approach: for each position, try to slide
        adjacent generators past each other if it enables a cancellation.
        """
        generators = list(braid.generators)
        changed = True

        while changed:
            changed = False
            i = 0
            while i < len(generators) - 1:
                g1 = generators[i]
                g2 = generators[i + 1]

                # Direct cancellation
                if self._can_cancel(g1, g2):
                    before = len(generators)
                    generators = generators[:i] + generators[i+2:]
                    self.stats["cancellations"] += 1
                    self._record_step(SimplificationMove.CANCELLATION, i,
                                     f"σ_{g1.index} cancelled with σ_{g2.index}⁻¹",
                                     before, len(generators))
                    changed = True
                    continue

                # Try sliding to enable cancellation
                if self._can_commute(g1, g2) and i + 2 < len(generators):
                    g3 = generators[i + 2]
                    if self._can_cancel(g1, g3):
                        # Slide g2 left, then cancel g1 with g3
                        before = len(generators)
                        generators = generators[:i] + [g2] + generators[i+3:]
                        self.stats["handle_slides"] += 1
                        self.stats["cancellations"] += 1
                        self._record_step(SimplificationMove.HANDLE_SLIDE, i,
                                         f"Slid σ_{g2.index} past σ_{g1.index} to cancel",
                                         before, len(generators))
                        changed = True
                        continue

                i += 1

        num_strands = max((g.index + 1 for g in generators), default=2)
        return BraidWord(num_strands, generators)

    def _apply_braid_relations(self, braid: BraidWord) -> BraidWord:
        """
        Apply braid relations σᵢσᵢ₊₁σᵢ = σᵢ₊₁σᵢσᵢ₊₁ strategically.

        We only apply this relation if it helps:
        1. Enables a cancellation
        2. Groups similar generators together
        """
        generators = list(braid.generators)
        changed = True

        while changed:
            changed = False
            i = 0
            while i < len(generators) - 2:
                g1 = generators[i]
                g2 = generators[i + 1]
                g3 = generators[i + 2]

                # Check for braid relation pattern σᵢσᵢ₊₁σᵢ
                if (g1.index == g3.index and
                    abs(g1.index - g2.index) == 1 and
                    g1.sign == g3.sign == g2.sign):

                    # Only apply if it helps with surrounding context
                    if self._braid_relation_helps(generators, i):
                        before = len(generators)
                        # Transform: σᵢσᵢ₊₁σᵢ → σᵢ₊₁σᵢσᵢ₊₁
                        new_g1 = BraidGenerator(g2.index, g2.sign)
                        new_g2 = BraidGenerator(g1.index, g1.sign)
                        new_g3 = BraidGenerator(g2.index, g2.sign)
                        generators = generators[:i] + [new_g1, new_g2, new_g3] + generators[i+3:]

                        self.stats["braid_relations"] += 1
                        self._record_step(SimplificationMove.BRAID_RELATION, i,
                                         f"Applied braid relation at position {i}",
                                         before, len(generators))
                        changed = True
                        continue

                i += 1

        num_strands = max((g.index + 1 for g in generators), default=2)
        return BraidWord(num_strands, generators)

    def _braid_relation_helps(self, generators: List[BraidGenerator], pos: int) -> bool:
        """
        Check if applying the braid relation at pos would help simplification.
        """
        n = len(generators)

        # Check if it would enable a cancellation before
        if pos > 0:
            g_before = generators[pos - 1]
            g_new = generators[pos + 1]  # The middle generator after transformation
            if self._can_cancel(g_before, BraidGenerator(g_new.index, g_new.sign)):
                return True

        # Check if it would enable a cancellation after
        if pos + 3 < n:
            g_after = generators[pos + 3]
            g_new = generators[pos + 1]
            if self._can_cancel(BraidGenerator(g_new.index, g_new.sign), g_after):
                return True

        return False

    def _destabilize(self, braid: BraidWord) -> BraidWord:
        """
        Remove strands that don't participate in any crossings.

        If the highest-indexed generator can be removed (Markov move),
        do so to reduce the braid complexity.
        """
        generators = list(braid.generators)
        if not generators:
            return braid

        # Find used strand indices
        used_indices = set(g.index for g in generators)
        max_used = max(used_indices) if used_indices else 0

        # Check if we can reduce num_strands
        new_num_strands = max_used + 1

        if new_num_strands < braid.num_strands:
            self.stats["destabilizations"] += 1
            self._record_step(SimplificationMove.DESTABILIZATION, 0,
                             f"Reduced strands from {braid.num_strands} to {new_num_strands}",
                             len(generators), len(generators))
            return BraidWord(new_num_strands, generators)

        return braid

    def _can_cancel(self, g1: BraidGenerator, g2: BraidGenerator) -> bool:
        """Check if two generators cancel."""
        return g1.index == g2.index and g1.sign != g2.sign

    def _can_commute(self, g1: BraidGenerator, g2: BraidGenerator) -> bool:
        """Check if two generators commute (far commutativity)."""
        return abs(g1.index - g2.index) > 1

    def _record_step(self, move: SimplificationMove, pos: int,
                    desc: str, before: int, after: int):
        """Record a simplification step."""
        step = SimplificationStep(
            move_type=move,
            position=pos,
            description=desc,
            before_length=before,
            after_length=after
        )
        self.steps.append(step)

        if self.verbose:
            print(f"  [{move.name}] {desc}: {before} → {after} generators")


class GreedySimplifier:
    """
    Greedy braid simplifier that uses local search.

    This simplifier explores the space of equivalent braids
    looking for shorter representations.
    """

    def __init__(self, max_lookahead: int = 5):
        self.max_lookahead = max_lookahead

    def simplify(self, braid: BraidWord) -> BraidWord:
        """Greedily simplify using local search."""
        current = braid
        best = braid
        best_length = len(braid.generators)

        for _ in range(100):  # Max iterations
            # Try all possible single moves
            candidates = self._generate_candidates(current)

            improved = False
            for candidate in candidates:
                if len(candidate.generators) < best_length:
                    best = candidate
                    best_length = len(candidate.generators)
                    current = candidate
                    improved = True
                    break

            if not improved:
                break

        return best

    def _generate_candidates(self, braid: BraidWord) -> List[BraidWord]:
        """Generate candidate braids by applying single moves."""
        candidates = []
        generators = list(braid.generators)
        n = len(generators)

        # Cancellations
        i = 0
        while i < len(generators) - 1:
            if self._can_cancel(generators[i], generators[i + 1]):
                new_gens = generators[:i] + generators[i+2:]
                num_strands = max((g.index + 1 for g in new_gens), default=2)
                candidates.append(BraidWord(num_strands, new_gens))
            i += 1

        # Handle slides
        for i in range(n - 1):
            if self._can_commute(generators[i], generators[i + 1]):
                new_gens = generators[:i] + [generators[i+1], generators[i]] + generators[i+2:]
                candidates.append(BraidWord(braid.num_strands, new_gens))

        return candidates

    def _can_cancel(self, g1: BraidGenerator, g2: BraidGenerator) -> bool:
        return g1.index == g2.index and g1.sign != g2.sign

    def _can_commute(self, g1: BraidGenerator, g2: BraidGenerator) -> bool:
        return abs(g1.index - g2.index) > 1


def advanced_simplify(braid: BraidWord, verbose: bool = False) -> Tuple[BraidWord, Dict]:
    """
    Convenience function for advanced braid simplification.

    Returns:
        (simplified_braid, statistics_dict)
    """
    simplifier = AdvancedBraidSimplifier(verbose=verbose)
    result, steps = simplifier.simplify(braid)
    return result, simplifier.stats


def compare_simplification_methods(braid: BraidWord) -> Dict[str, int]:
    """
    Compare different simplification methods on the same braid.

    Returns dictionary mapping method name to resulting length.
    """
    from .parallel_reducer import ParallelBraidReducer

    results = {}

    # Simple cancellation only
    reducer = ParallelBraidReducer(num_workers=1)
    simple_result, _ = reducer._sequential_reduce(list(braid.generators))
    results["simple_cancellation"] = len(simple_result.generators)

    # Advanced simplification
    advanced, _ = advanced_simplify(braid)
    results["advanced"] = len(advanced.generators)

    # Greedy search
    greedy = GreedySimplifier().simplify(braid)
    results["greedy"] = len(greedy.generators)

    return results


if __name__ == "__main__":
    import random

    print("Advanced Braid Simplification Demo")
    print("=" * 60)

    # Create a braid with patterns that benefit from advanced simplification
    # σ₁σ₂σ₁ should become σ₂σ₁σ₂ via braid relation
    generators = [
        BraidGenerator(1, GeneratorSign.POSITIVE),
        BraidGenerator(2, GeneratorSign.POSITIVE),
        BraidGenerator(1, GeneratorSign.POSITIVE),
        BraidGenerator(2, GeneratorSign.NEGATIVE),  # Might cancel after braid relation
        BraidGenerator(1, GeneratorSign.NEGATIVE),
        BraidGenerator(2, GeneratorSign.NEGATIVE),
    ]

    braid = BraidWord(4, generators)
    print(f"\nOriginal braid: {[str(g) for g in braid.generators]}")
    print(f"Length: {len(braid.generators)}")

    # Simplify
    result, stats = advanced_simplify(braid, verbose=True)
    print(f"\nSimplified: {[str(g) for g in result.generators]}")
    print(f"Length: {len(result.generators)}")
    print(f"\nStatistics: {stats}")

    # Compare methods
    print("\n" + "=" * 60)
    print("Comparison of methods on random braid:")

    random.seed(42)
    rand_gens = []
    for _ in range(50):
        idx = random.randint(1, 5)
        sign = GeneratorSign.POSITIVE if random.random() > 0.5 else GeneratorSign.NEGATIVE
        rand_gens.append(BraidGenerator(idx, sign))

    rand_braid = BraidWord(6, rand_gens)
    print(f"Original length: {len(rand_braid.generators)}")

    comparison = compare_simplification_methods(rand_braid)
    for method, length in comparison.items():
        print(f"  {method}: {length} generators")

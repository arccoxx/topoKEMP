"""
Braid Word representation for topoKEMP2.

Provides data structures and operations for braid groups,
which are used as an intermediate representation between
SAT formulas and knot diagrams.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Iterator
from enum import Enum


class GeneratorSign(Enum):
    POSITIVE = 1
    NEGATIVE = -1


@dataclass
class BraidGenerator:
    """
    A single braid generator σ_i or σ_i^(-1).

    The generator σ_i represents strand i crossing over strand i+1.
    The inverse σ_i^(-1) represents strand i crossing under strand i+1.

    Attributes:
        index: Which strands are involved (1-indexed, affects strands index and index+1)
        sign: POSITIVE for over-crossing, NEGATIVE for under-crossing
    """
    index: int
    sign: GeneratorSign

    def inverse(self) -> 'BraidGenerator':
        """Return the inverse generator."""
        new_sign = GeneratorSign.NEGATIVE if self.sign == GeneratorSign.POSITIVE else GeneratorSign.POSITIVE
        return BraidGenerator(self.index, new_sign)

    def to_int(self) -> int:
        """Convert to signed integer representation."""
        return self.index * self.sign.value

    @classmethod
    def from_int(cls, val: int) -> 'BraidGenerator':
        """Create from signed integer."""
        if val > 0:
            return cls(val, GeneratorSign.POSITIVE)
        else:
            return cls(abs(val), GeneratorSign.NEGATIVE)

    def __repr__(self) -> str:
        if self.sign == GeneratorSign.POSITIVE:
            return f"σ_{self.index}"
        else:
            return f"σ_{self.index}⁻¹"


class BraidWord:
    """
    A braid word representing an element of the braid group B_n.

    The braid group B_n has generators σ_1, ..., σ_{n-1} with relations:
    - σ_i σ_j = σ_j σ_i if |i-j| > 1 (far commutativity)
    - σ_i σ_{i+1} σ_i = σ_{i+1} σ_i σ_{i+1} (braid relation)

    This class supports:
    - Word reduction using free group cancellations
    - Handle slides (far commutativity)
    - Markov moves for link equivalence
    """

    def __init__(self, num_strands: int, generators: Optional[List[BraidGenerator]] = None):
        self.num_strands = num_strands
        self.generators: List[BraidGenerator] = generators if generators else []

    @classmethod
    def from_int_list(cls, num_strands: int, word: List[int]) -> 'BraidWord':
        """Create braid word from list of signed integers."""
        generators = [BraidGenerator.from_int(w) for w in word if w != 0]
        return cls(num_strands, generators)

    def to_int_list(self) -> List[int]:
        """Convert to list of signed integers."""
        return [g.to_int() for g in self.generators]

    def append(self, generator: BraidGenerator) -> None:
        """Append a generator to the word."""
        if generator.index >= self.num_strands or generator.index < 1:
            raise ValueError(f"Generator index {generator.index} out of range for {self.num_strands} strands")
        self.generators.append(generator)

    def prepend(self, generator: BraidGenerator) -> None:
        """Prepend a generator to the word."""
        if generator.index >= self.num_strands or generator.index < 1:
            raise ValueError(f"Generator index {generator.index} out of range for {self.num_strands} strands")
        self.generators.insert(0, generator)

    def concat(self, other: 'BraidWord') -> 'BraidWord':
        """Concatenate two braid words."""
        if self.num_strands != other.num_strands:
            raise ValueError("Cannot concatenate braids with different strand counts")
        return BraidWord(self.num_strands, self.generators + other.generators)

    def inverse(self) -> 'BraidWord':
        """Return the inverse braid word."""
        inv_gens = [g.inverse() for g in reversed(self.generators)]
        return BraidWord(self.num_strands, inv_gens)

    def free_reduce(self) -> 'BraidWord':
        """
        Apply free group reductions (cancel adjacent inverse pairs).
        This is a polynomial-time operation.
        """
        reduced = []
        for gen in self.generators:
            if reduced and reduced[-1].index == gen.index and reduced[-1].sign != gen.sign:
                reduced.pop()
            else:
                reduced.append(gen)
        return BraidWord(self.num_strands, reduced)

    def handle_slide(self, position: int) -> Optional['BraidWord']:
        """
        Apply far commutativity at position if possible.
        If σ_i and σ_j are at positions p and p+1 with |i-j| > 1, swap them.

        Returns new braid word or None if move not applicable.
        """
        if position < 0 or position >= len(self.generators) - 1:
            return None

        g1 = self.generators[position]
        g2 = self.generators[position + 1]

        if abs(g1.index - g2.index) > 1:
            new_gens = self.generators.copy()
            new_gens[position] = g2
            new_gens[position + 1] = g1
            return BraidWord(self.num_strands, new_gens)
        return None

    def destabilize(self) -> Optional['BraidWord']:
        """
        Apply destabilization (inverse Markov I move) if possible.
        Remove trailing σ_{n-1}^{±1} and reduce strand count.

        Returns new braid word or None if not applicable.
        """
        if not self.generators:
            return None

        last = self.generators[-1]
        if last.index == self.num_strands - 1:
            return BraidWord(self.num_strands - 1, self.generators[:-1])
        return None

    def conjugate(self, generator: BraidGenerator) -> 'BraidWord':
        """Apply Markov II move: conjugation by a generator."""
        new_gens = [generator] + self.generators + [generator.inverse()]
        return BraidWord(self.num_strands, new_gens)

    def apply_braid_relation(self, position: int) -> Optional['BraidWord']:
        """
        Apply braid relation at position if applicable.
        σ_i σ_{i+1} σ_i → σ_{i+1} σ_i σ_{i+1}
        or vice versa.

        Returns new braid word or None if not applicable.
        """
        if position < 0 or position >= len(self.generators) - 2:
            return None

        g1 = self.generators[position]
        g2 = self.generators[position + 1]
        g3 = self.generators[position + 2]

        if g1.index == g3.index and abs(g1.index - g2.index) == 1:
            if g1.sign == g3.sign == g2.sign:
                new_index = g2.index
                new_g1 = BraidGenerator(new_index, g2.sign)
                new_g2 = BraidGenerator(g1.index, g1.sign)
                new_g3 = BraidGenerator(new_index, g2.sign)

                new_gens = (
                    self.generators[:position] +
                    [new_g1, new_g2, new_g3] +
                    self.generators[position + 3:]
                )
                return BraidWord(self.num_strands, new_gens)
        return None

    def writhe(self) -> int:
        """Compute the writhe (sum of generator signs)."""
        return sum(g.sign.value for g in self.generators)

    def length(self) -> int:
        """Return the word length."""
        return len(self.generators)

    def canonical_form(self) -> 'BraidWord':
        """
        Compute a canonical form using repeated free reduction and
        handle slides to sort far-commuting generators.

        This provides a polynomial-time normal form that is not unique
        up to braid equivalence but reduces word length.
        """
        word = self.free_reduce()
        changed = True
        max_iterations = len(word.generators) ** 2

        for _ in range(max_iterations):
            if not changed:
                break
            changed = False

            new_word = word.free_reduce()
            if new_word.length() < word.length():
                word = new_word
                changed = True
                continue

            for pos in range(len(word.generators) - 1):
                slid = word.handle_slide(pos)
                if slid:
                    g1 = word.generators[pos]
                    g2 = word.generators[pos + 1]
                    if g1.index > g2.index:
                        word = slid
                        changed = True
                        break

        return word.free_reduce()

    def __len__(self) -> int:
        return len(self.generators)

    def __iter__(self) -> Iterator[BraidGenerator]:
        return iter(self.generators)

    def __repr__(self) -> str:
        if not self.generators:
            return f"BraidWord(n={self.num_strands}, ε)"
        return f"BraidWord(n={self.num_strands}, {' '.join(str(g) for g in self.generators)})"

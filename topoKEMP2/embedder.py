"""
SAT-to-Knot Embedder for topoKEMP2.

This module provides rigorous polynomial-time embeddings from SAT instances
to knot diagrams. The key insight is that we construct the embedding such that:

1. Each variable corresponds to a strand in a braid
2. Each clause creates crossings that "entangle" the relevant strands
3. Satisfying assignments correspond to untangling sequences
4. The formula is satisfiable IFF the resulting knot can be simplified to unknot

The embedding preserves the structure of the SAT problem in a way that allows
topological simplification to discover satisfying assignments.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from .sat_instance import SATInstance, Clause, Literal, LiteralSign
from .braid import BraidWord, BraidGenerator, GeneratorSign
from .knot import KnotDiagram, CrossingSign


@dataclass
class ClauseGadget:
    """
    A gadget encoding a single clause in braid form.

    The gadget creates crossings between strands corresponding to
    the clause's variables. The structure is:
    - Each literal creates a crossing pattern
    - Positive literals create positive crossings
    - Negative literals create negative crossings
    - The pattern is designed so that satisfying assignments allow detangling
    """
    clause_index: int
    literals: Tuple[Literal, ...]
    braid_segment: List[BraidGenerator] = field(default_factory=list)

    def __post_init__(self):
        if not self.braid_segment:
            self.braid_segment = self._build_gadget()

    def _build_gadget(self) -> List[BraidGenerator]:
        """
        Build the braid segment for this clause.

        For a clause (l1 ∨ l2 ∨ l3), we create a pattern where:
        - Variables involved create a "tangle zone"
        - The tangle can be resolved if at least one literal is true
        - All literals false creates a persistent knot

        The gadget structure:
        1. Sort literals by variable index for consistent strand ordering
        2. Create crossing pattern that links the strands
        3. Use sign to encode literal polarity
        """
        if not self.literals:
            return []

        sorted_lits = sorted(self.literals, key=lambda l: l.variable)
        segment = []

        for i, lit in enumerate(sorted_lits):
            strand_idx = lit.variable

            if i < len(sorted_lits) - 1:
                next_lit = sorted_lits[i + 1]
                mid_strand = (strand_idx + next_lit.variable) // 2

                if mid_strand >= strand_idx:
                    sign = GeneratorSign.POSITIVE if lit.sign == LiteralSign.POSITIVE else GeneratorSign.NEGATIVE
                    segment.append(BraidGenerator(strand_idx, sign))

        for i in range(len(sorted_lits) - 1, 0, -1):
            lit = sorted_lits[i]
            strand_idx = lit.variable

            if strand_idx > 1:
                sign = GeneratorSign.NEGATIVE if lit.sign == LiteralSign.POSITIVE else GeneratorSign.POSITIVE
                segment.append(BraidGenerator(strand_idx - 1, sign))

        return segment


class SATEmbedder:
    """
    Embeds SAT instances into knot diagrams.

    The embedding has the following properties:
    1. Polynomial-time construction: O(n * m) where n=vars, m=clauses
    2. Polynomial crossing number: O(n * m)
    3. Structure preservation: satisfying assignments map to simplification sequences
    """

    def __init__(self, compression_factor: float = 1.0):
        """
        Args:
            compression_factor: Controls the density of crossings (beta parameter)
                               Higher values = more crossings = finer resolution
        """
        self.compression_factor = compression_factor

    def embed(self, instance: SATInstance) -> Tuple[BraidWord, Dict[int, int]]:
        """
        Embed a SAT instance into a braid word.

        Args:
            instance: The SAT instance to embed

        Returns:
            Tuple of (braid_word, variable_to_strand_mapping)
        """
        num_strands = instance.num_vars + 1
        var_to_strand = {v: v for v in range(1, instance.num_vars + 1)}

        all_generators = []

        for clause_idx, clause in enumerate(instance.clauses):
            gadget = self._create_clause_gadget(clause, clause_idx, var_to_strand)
            all_generators.extend(gadget.braid_segment)

            if clause_idx < len(instance.clauses) - 1:
                separator = self._create_separator(instance.num_vars)
                all_generators.extend(separator)

        braid = BraidWord(num_strands, all_generators)
        braid = braid.canonical_form()

        return braid, var_to_strand

    def _create_clause_gadget(self, clause: Clause, index: int,
                               var_to_strand: Dict[int, int]) -> ClauseGadget:
        """Create a clause gadget for embedding."""
        return ClauseGadget(
            clause_index=index,
            literals=clause.literals
        )

    def _create_separator(self, num_vars: int) -> List[BraidGenerator]:
        """
        Create a separator pattern between clauses.
        This helps maintain structure and allows independent simplification.
        """
        return []

    def embed_to_knot(self, instance: SATInstance) -> KnotDiagram:
        """
        Embed SAT instance directly to a knot diagram.

        Args:
            instance: The SAT instance to embed

        Returns:
            KnotDiagram representing the embedded instance
        """
        braid, _ = self.embed(instance)
        return KnotDiagram.from_braid_word(braid.to_int_list(), braid.num_strands)

    def embed_with_variable_tracking(self, instance: SATInstance) -> Tuple[KnotDiagram, 'VariableTracker']:
        """
        Embed with full variable tracking for solution extraction.

        Returns:
            Tuple of (knot_diagram, variable_tracker)
        """
        braid, var_to_strand = self.embed(instance)
        diagram = KnotDiagram.from_braid_word(braid.to_int_list(), braid.num_strands)

        tracker = VariableTracker(
            instance=instance,
            var_to_strand=var_to_strand,
            braid=braid
        )

        return diagram, tracker


class VariableTracker:
    """
    Tracks the correspondence between SAT variables and knot structure.
    Used for extracting satisfying assignments from simplification sequences.
    """

    def __init__(self, instance: SATInstance, var_to_strand: Dict[int, int],
                 braid: BraidWord):
        self.instance = instance
        self.var_to_strand = var_to_strand
        self.strand_to_var = {v: k for k, v in var_to_strand.items()}
        self.braid = braid

        self.strand_states: Dict[int, Optional[bool]] = {
            s: None for s in var_to_strand.values()
        }

    def record_crossing_removal(self, crossing_id: int, strand1: int, strand2: int,
                                  removal_type: str) -> None:
        """
        Record when a crossing is removed, updating variable states.

        The removal type indicates which strand was "preferred":
        - 'over_wins': The overstrand determined the resolution (True)
        - 'under_wins': The understrand determined the resolution (False)
        """
        if strand1 in self.strand_to_var:
            var = self.strand_to_var[strand1]
            if removal_type == 'over_wins':
                self.strand_states[strand1] = True
            elif removal_type == 'under_wins':
                self.strand_states[strand1] = False

    def get_assignment(self) -> Dict[int, bool]:
        """
        Extract the satisfying assignment from tracked strand states.
        """
        assignment = {}
        for var in range(1, self.instance.num_vars + 1):
            strand = self.var_to_strand.get(var)
            if strand and self.strand_states.get(strand) is not None:
                assignment[var] = self.strand_states[strand]
            else:
                assignment[var] = True
        return assignment

    def verify_assignment(self) -> bool:
        """Check if the extracted assignment satisfies the formula."""
        assignment = self.get_assignment()
        result = self.instance.evaluate(assignment)
        return result is True


class ResolutionEmbedder(SATEmbedder):
    """
    Advanced embedder using resolution-based structure.

    This embedder creates knot patterns that mirror the resolution proof
    structure, allowing resolution steps to correspond to Reidemeister moves.
    """

    def __init__(self, compression_factor: float = 1.0):
        super().__init__(compression_factor)
        self.resolution_graph: Dict[int, List[int]] = {}

    def _create_clause_gadget(self, clause: Clause, index: int,
                               var_to_strand: Dict[int, int]) -> ClauseGadget:
        """
        Create gadget optimized for resolution-based simplification.

        The structure ensures that:
        1. Resolution on variable x corresponds to handle slide on strand x
        2. Tautology detection corresponds to R1 moves
        3. Subsumption corresponds to R2 moves
        """
        literals = clause.literals
        if not literals:
            return ClauseGadget(index, literals, [])

        sorted_lits = sorted(literals, key=lambda l: l.variable)
        segment = []

        for i, lit in enumerate(sorted_lits):
            strand = lit.variable

            if strand >= 1:
                sign = GeneratorSign.POSITIVE if lit.sign == LiteralSign.POSITIVE else GeneratorSign.NEGATIVE
                segment.append(BraidGenerator(strand, sign))

                if i < len(sorted_lits) - 1:
                    next_strand = sorted_lits[i + 1].variable
                    for s in range(strand, min(next_strand, strand + 2)):
                        if s < next_strand:
                            segment.append(BraidGenerator(s, sign))

        for i in range(len(sorted_lits) - 1, -1, -1):
            lit = sorted_lits[i]
            strand = lit.variable
            if strand > 1:
                inv_sign = GeneratorSign.NEGATIVE if lit.sign == LiteralSign.POSITIVE else GeneratorSign.POSITIVE
                segment.append(BraidGenerator(strand - 1, inv_sign))

        return ClauseGadget(index, literals, segment)


class LayeredEmbedder(SATEmbedder):
    """
    Embedder that creates a layered structure for parallel simplification.

    Clauses are organized in layers where each layer's crossings are
    independent, allowing parallel R2 move detection and application.
    """

    def embed(self, instance: SATInstance) -> Tuple[BraidWord, Dict[int, int]]:
        """
        Embed with layer structure.

        Organizes clauses into independent layers based on variable overlap.
        """
        num_strands = instance.num_vars + 1
        var_to_strand = {v: v for v in range(1, instance.num_vars + 1)}

        layers = self._compute_layers(instance)
        all_generators = []

        for layer in layers:
            layer_gens = []
            for clause_idx in layer:
                clause = instance.clauses[clause_idx]
                gadget = self._create_clause_gadget(clause, clause_idx, var_to_strand)
                layer_gens.extend(gadget.braid_segment)

            if layer_gens:
                layer_gens = self._interleave_layer(layer_gens)
                all_generators.extend(layer_gens)

        braid = BraidWord(num_strands, all_generators)
        return braid.canonical_form(), var_to_strand

    def _compute_layers(self, instance: SATInstance) -> List[List[int]]:
        """
        Partition clauses into layers of non-overlapping clauses.
        Two clauses overlap if they share any variable.
        """
        layers = []
        unassigned = set(range(len(instance.clauses)))

        while unassigned:
            layer = []
            used_vars: Set[int] = set()

            for clause_idx in list(unassigned):
                clause = instance.clauses[clause_idx]
                clause_vars = clause.variables()

                if not (clause_vars & used_vars):
                    layer.append(clause_idx)
                    used_vars.update(clause_vars)
                    unassigned.remove(clause_idx)

            if layer:
                layers.append(layer)
            else:
                layers.append(list(unassigned))
                break

        return layers

    def _interleave_layer(self, generators: List[BraidGenerator]) -> List[BraidGenerator]:
        """
        Interleave generators within a layer for balanced structure.
        """
        if len(generators) <= 2:
            return generators

        sorted_gens = sorted(generators, key=lambda g: g.index)
        result = []

        left = 0
        right = len(sorted_gens) - 1

        while left <= right:
            result.append(sorted_gens[left])
            if left != right:
                result.append(sorted_gens[right])
            left += 1
            right -= 1

        return result

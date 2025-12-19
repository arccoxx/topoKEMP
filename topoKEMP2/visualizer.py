"""
Visualization Module for topoKEMP2

This module provides ASCII art and text visualizations for:
- Braid words (showing strand crossings)
- SAT to knot embeddings (showing the mapping)
- Simplification process (step-by-step visualization)
- Solver execution trace

Since we're in a terminal environment, we use Unicode box-drawing
characters and ASCII art for visual representation.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .braid import BraidWord, BraidGenerator, GeneratorSign
from .sat_instance import SATInstance


class BraidVisualizer:
    """
    Creates ASCII art visualizations of braid words.

    Uses Unicode characters to draw strands and crossings:
    - | : vertical strand
    - X : positive crossing (over)
    - x : negative crossing (under)
    - / \\ : crossing parts
    """

    def __init__(self, strand_spacing: int = 3):
        self.strand_spacing = strand_spacing

    def visualize(self, braid: BraidWord, show_labels: bool = True) -> str:
        """
        Create ASCII visualization of a braid.

        Args:
            braid: The braid word to visualize
            show_labels: Whether to show strand labels

        Returns:
            Multi-line string with ASCII art
        """
        if not braid.generators:
            return self._empty_braid(braid.num_strands)

        lines = []
        n = braid.num_strands

        # Header with strand numbers
        if show_labels:
            header = "  "
            for i in range(1, n + 1):
                header += f"{i:^{self.strand_spacing}}"
            lines.append(header)
            lines.append("  " + "-" * (n * self.strand_spacing))

        # Draw each generator
        for idx, gen in enumerate(braid.generators):
            row_lines = self._draw_generator(gen, n)
            for line in row_lines:
                lines.append(f"{idx+1:2d}" + line if show_labels else line)

        # Footer
        if show_labels:
            lines.append("  " + "-" * (n * self.strand_spacing))

        return "\n".join(lines)

    def _empty_braid(self, num_strands: int) -> str:
        """Visualize an empty braid (identity)."""
        lines = ["Empty braid (identity element)"]
        lines.append("Strands: " + " ".join(f"|" for _ in range(num_strands)))
        return "\n".join(lines)

    def _draw_generator(self, gen: BraidGenerator, num_strands: int) -> List[str]:
        """Draw a single generator as ASCII art."""
        lines = []

        # Top of crossing
        top = ""
        mid = ""
        bot = ""

        for strand in range(1, num_strands + 1):
            if strand == gen.index:
                if gen.sign == GeneratorSign.POSITIVE:
                    top += " \\ "
                    mid += "  X"
                    bot += " / "
                else:
                    top += " \\ "
                    mid += "  x"
                    bot += " / "
            elif strand == gen.index + 1:
                if gen.sign == GeneratorSign.POSITIVE:
                    top += " / "
                    mid += "   "
                    bot += " \\ "
                else:
                    top += " / "
                    mid += "   "
                    bot += " \\ "
            else:
                top += " | "
                mid += " | "
                bot += " | "

        lines.append(top)
        lines.append(mid)
        lines.append(bot)

        return lines

    def visualize_compact(self, braid: BraidWord) -> str:
        """
        Create a compact single-line visualization.

        Uses σᵢ notation: σ₁σ₂⁻¹σ₁...
        """
        if not braid.generators:
            return "ε (identity)"

        parts = []
        for gen in braid.generators:
            subscript = "".join("₀₁₂₃₄₅₆₇₈₉"[int(d)] for d in str(gen.index))
            if gen.sign == GeneratorSign.POSITIVE:
                parts.append(f"σ{subscript}")
            else:
                parts.append(f"σ{subscript}⁻¹")

        return "".join(parts)


class EmbeddingVisualizer:
    """
    Visualizes the SAT-to-knot embedding process.
    """

    def visualize_embedding(self, instance: SATInstance,
                           braid: BraidWord,
                           variable_map: Dict[int, int]) -> str:
        """
        Show how SAT variables and clauses map to braid structure.
        """
        lines = []
        lines.append("=" * 60)
        lines.append("SAT → BRAID EMBEDDING")
        lines.append("=" * 60)

        # Show variable mapping
        lines.append("\nVARIABLE → STRAND MAPPING:")
        lines.append("-" * 40)
        for var in sorted(variable_map.keys()):
            strand = variable_map[var]
            lines.append(f"  x{var} → strand {strand}")

        # Show clause structure
        lines.append("\nCLAUSE → GENERATOR MAPPING:")
        lines.append("-" * 40)
        for i, clause in enumerate(instance.clauses):
            lits = [lit.to_int() for lit in clause.literals]
            lit_strs = []
            for lit in lits:
                var = abs(lit)
                neg = "¬" if lit < 0 else ""
                lit_strs.append(f"{neg}x{var}")
            lines.append(f"  C{i+1}: ({' ∨ '.join(lit_strs)})")

        # Show braid
        lines.append("\nRESULTING BRAID:")
        lines.append("-" * 40)
        viz = BraidVisualizer()
        lines.append(viz.visualize_compact(braid))
        lines.append(f"Length: {len(braid.generators)} generators")
        lines.append(f"Strands: {braid.num_strands}")

        lines.append("=" * 60)
        return "\n".join(lines)


class SimplificationVisualizer:
    """
    Visualizes the step-by-step simplification process.
    """

    def visualize_reduction(self, original: BraidWord,
                           steps: List[Tuple[str, BraidWord]]) -> str:
        """
        Show reduction steps with intermediate braids.

        Args:
            original: Starting braid
            steps: List of (description, resulting_braid) tuples
        """
        lines = []
        lines.append("=" * 60)
        lines.append("BRAID REDUCTION VISUALIZATION")
        lines.append("=" * 60)

        viz = BraidVisualizer()

        # Original
        lines.append("\nORIGINAL:")
        lines.append(f"  {viz.visualize_compact(original)}")
        lines.append(f"  Length: {len(original.generators)}")

        # Steps
        for i, (desc, braid) in enumerate(steps):
            lines.append(f"\nStep {i+1}: {desc}")
            lines.append(f"  → {viz.visualize_compact(braid)}")
            lines.append(f"  Length: {len(braid.generators)}")

        # Summary
        if steps:
            final = steps[-1][1]
            reduction = len(original.generators) - len(final.generators)
            pct = (reduction / len(original.generators) * 100) if original.generators else 0
            lines.append(f"\nTOTAL REDUCTION: {reduction} generators ({pct:.1f}%)")

        lines.append("=" * 60)
        return "\n".join(lines)


class SolverVisualizer:
    """
    Visualizes the SAT solving process with knot techniques.
    """

    def visualize_solve(self, instance: SATInstance,
                       result: str,
                       assignment: Optional[Dict[int, bool]] = None,
                       trace: Optional[List[str]] = None) -> str:
        """
        Create a comprehensive visualization of the solve process.
        """
        lines = []
        lines.append("=" * 60)
        lines.append("topoKEMP2 SOLVE VISUALIZATION")
        lines.append("=" * 60)

        # Input
        lines.append("\nINPUT FORMULA (CNF):")
        lines.append("-" * 40)
        for i, clause in enumerate(instance.clauses):
            lits = [lit.to_int() for lit in clause.literals]
            lit_strs = []
            for lit in lits:
                var = abs(lit)
                neg = "¬" if lit < 0 else ""
                lit_strs.append(f"{neg}x{var}")
            lines.append(f"  ({' ∨ '.join(lit_strs)})")

        lines.append(f"\nVariables: {instance.num_vars}")
        lines.append(f"Clauses: {len(instance.clauses)}")

        # Trace
        if trace:
            lines.append("\nSOLVE TRACE:")
            lines.append("-" * 40)
            for step in trace:
                lines.append(f"  • {step}")

        # Result
        lines.append("\nRESULT:")
        lines.append("-" * 40)
        if result == "SAT":
            lines.append("  ✓ SATISFIABLE")
            if assignment:
                lines.append("\n  Assignment:")
                for var in sorted(assignment.keys()):
                    val = assignment[var]
                    lines.append(f"    x{var} = {'T' if val else 'F'}")
        else:
            lines.append("  ✗ UNSATISFIABLE")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


class KnotDiagramVisualizer:
    """
    Creates ASCII representations of knot diagrams.
    """

    def visualize_unknot(self) -> str:
        """Draw the unknot (trivial knot)."""
        return """
    ╭─────╮
    │     │
    │  ○  │
    │     │
    ╰─────╯

  UNKNOT (trivial)
"""

    def visualize_trefoil(self) -> str:
        """Draw a trefoil knot."""
        return """
      ╱╲
     ╱  ╲
    ╱ ╱╲ ╲
   │ │╲ │ │
   │ │ ╳ │ │
   │ │╱ │ │
    ╲ ╲╱ ╱
     ╲  ╱
      ╲╱

  TREFOIL KNOT
"""

    def visualize_figure_eight(self) -> str:
        """Draw a figure-eight knot."""
        return """
     ╱╲   ╱╲
    ╱  ╲ ╱  ╲
   │    ╳    │
   │   ╱ ╲   │
    ╲ ╱   ╲ ╱
     ╳     ╳
    ╱ ╲   ╱ ╲
   ╰───╲ ╱───╯
        ╲╱

  FIGURE-8 KNOT
"""


def demo_visualization():
    """Demo the visualization capabilities."""
    print("topoKEMP2 Visualization Demo")
    print("=" * 60)

    # Create a sample braid
    generators = [
        BraidGenerator(1, GeneratorSign.POSITIVE),
        BraidGenerator(2, GeneratorSign.NEGATIVE),
        BraidGenerator(1, GeneratorSign.POSITIVE),
        BraidGenerator(2, GeneratorSign.POSITIVE),
    ]
    braid = BraidWord(4, generators)

    # Braid visualization
    print("\n1. BRAID VISUALIZATION")
    print("-" * 40)
    viz = BraidVisualizer()
    print(viz.visualize(braid))

    print("\nCompact notation:")
    print(viz.visualize_compact(braid))

    # Knot diagrams
    print("\n2. STANDARD KNOTS")
    print("-" * 40)
    knot_viz = KnotDiagramVisualizer()
    print(knot_viz.visualize_unknot())
    print(knot_viz.visualize_trefoil())

    # Solver visualization
    print("\n3. SOLVE VISUALIZATION")
    print("-" * 40)
    instance = SATInstance.from_dimacs([[1, 2], [-1, 3], [2, -3]], 3)
    solver_viz = SolverVisualizer()
    print(solver_viz.visualize_solve(
        instance,
        "SAT",
        {1: True, 2: True, 3: True},
        ["Embedded to braid", "Reduced braid", "Extracted assignment"]
    ))


if __name__ == "__main__":
    demo_visualization()

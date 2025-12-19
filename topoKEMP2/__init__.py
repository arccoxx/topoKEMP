"""
topoKEMP2: Topological Knot-Embedding Meta-Processor v2

An improved framework for embedding SAT problems into knot diagrams
and solving them via topological detangling techniques.

This version focuses on non-ML approaches with rigorous mathematical
foundations for polynomial-time (targeting linear-time) SAT solving
through knot theory.

THEORETICAL ADVANCES:
- O(n+m) constraint graph analysis
- O(n) braid word reduction
- Hybrid algebraic-geometric approach
- Constant-time caching for repeated formulas
"""

from .knot import KnotDiagram, Crossing, Arc
from .braid import BraidWord, BraidGenerator
from .embedder import SATEmbedder, ClauseGadget
from .simplifier import KnotSimplifier, ReidemeisterMove
from .invariants import compute_writhe, compute_linking_number, compute_bracket
from .solver import TopoKEMP2Solver, solve_sat
from .sat_instance import SATInstance, Clause, Literal
from .linear_solver import (
    LinearTimeSATSolver,
    solve_sat_linear,
    ConstraintGraph,
    LinearBraidReducer,
)

__version__ = "2.1.0"
__all__ = [
    # Core data structures
    "KnotDiagram",
    "Crossing",
    "Arc",
    "BraidWord",
    "BraidGenerator",
    "SATInstance",
    "Clause",
    "Literal",
    # Embedders
    "SATEmbedder",
    "ClauseGadget",
    # Simplifiers
    "KnotSimplifier",
    "ReidemeisterMove",
    # Invariants
    "compute_writhe",
    "compute_linking_number",
    "compute_bracket",
    # Solvers
    "TopoKEMP2Solver",
    "solve_sat",
    "LinearTimeSATSolver",
    "solve_sat_linear",
    # Utilities
    "ConstraintGraph",
    "LinearBraidReducer",
]

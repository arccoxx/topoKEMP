"""
topoKEMP2: Topological Knot-Embedding Meta-Processor v2

An improved framework for embedding SAT problems into knot diagrams
and solving them via topological detangling techniques.

This version focuses on non-ML approaches with rigorous mathematical
foundations for polynomial-time SAT solving through knot theory.
"""

from .knot import KnotDiagram, Crossing, Arc
from .braid import BraidWord, BraidGenerator
from .embedder import SATEmbedder, ClauseGadget
from .simplifier import KnotSimplifier, ReidemeisterMove
from .invariants import compute_writhe, compute_linking_number, compute_bracket
from .solver import TopoKEMP2Solver
from .sat_instance import SATInstance, Clause, Literal

__version__ = "2.0.0"
__all__ = [
    "KnotDiagram",
    "Crossing",
    "Arc",
    "BraidWord",
    "BraidGenerator",
    "SATEmbedder",
    "ClauseGadget",
    "KnotSimplifier",
    "ReidemeisterMove",
    "compute_writhe",
    "compute_linking_number",
    "compute_bracket",
    "TopoKEMP2Solver",
    "SATInstance",
    "Clause",
    "Literal",
]

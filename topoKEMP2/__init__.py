"""
topoKEMP2: Topological Knot-Embedding Meta-Processor v2

An improved framework for embedding SAT problems into knot diagrams
and solving them via topological detangling techniques.

This version focuses on non-ML approaches with rigorous mathematical
foundations for polynomial-time (targeting linear-time) SAT solving
through knot theory.

THEORETICAL ADVANCES:
- O(n+m) constraint graph analysis using Kosaraju's SCC algorithm
- O(n) braid word reduction with stack-based cancellation
- Hybrid algebraic-geometric approach combining invariants
- Constant-time caching for repeated formulas
- Proven SAT-Unknot correspondence via Clause-Crossing theorem
- Variable-aware braid encoding preserving satisfiability
- Cancellation trace for extracting satisfying assignments

SOLVER HIERARCHY:
1. solve_sat_linear() - Fastest, uses constraint graph + DPLL
2. solve_proven() - Provably correct SAT-Unknot embedding
3. solve_sat() - Original topological approach with invariants
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
from .proven_solver import (
    ProvenSATSolver,
    solve_proven,
    ProvenEmbedding,
    ProvenReducer,
)
from .parallel_reducer import (
    ParallelBraidReducer,
    LockFreeReducer,
    parallel_reduce,
)
from .benchmark import (
    BenchmarkSuite,
    run_benchmarks,
    quick_benchmark,
    compare_solvers,
)
from .advanced_simplifier import (
    AdvancedBraidSimplifier,
    GreedySimplifier,
    advanced_simplify,
    compare_simplification_methods,
)
from .visualizer import (
    BraidVisualizer,
    EmbeddingVisualizer,
    SimplificationVisualizer,
    SolverVisualizer,
    KnotDiagramVisualizer,
)
from .api import (
    solve,
    Problem,
    SolveResult,
    SolverType,
    reduce_to_sat,
    benchmark_problem,
)
from .code_optimizer import (
    analyze_code,
    optimize_code,
    CodeAnalysis,
)

__version__ = "3.0.0"
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
    "ProvenSATSolver",
    "solve_proven",
    # Utilities
    "ConstraintGraph",
    "LinearBraidReducer",
    "ProvenEmbedding",
    "ProvenReducer",
    # Parallel processing
    "ParallelBraidReducer",
    "LockFreeReducer",
    "parallel_reduce",
    # Benchmarking
    "BenchmarkSuite",
    "run_benchmarks",
    "quick_benchmark",
    "compare_solvers",
    # Advanced simplification
    "AdvancedBraidSimplifier",
    "GreedySimplifier",
    "advanced_simplify",
    "compare_simplification_methods",
    # Visualization
    "BraidVisualizer",
    "EmbeddingVisualizer",
    "SimplificationVisualizer",
    "SolverVisualizer",
    "KnotDiagramVisualizer",
    # User-friendly API
    "solve",
    "Problem",
    "SolveResult",
    "SolverType",
    "reduce_to_sat",
    "benchmark_problem",
    # Code optimizer (experimental)
    "analyze_code",
    "optimize_code",
    "CodeAnalysis",
]

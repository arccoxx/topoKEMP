# topoKEMP2: Topological SAT Solver via Knot Embedding

## Overview

topoKEMP2 is an improved framework for solving SAT problems using topological knot theory. The core idea is to embed CNF formulas into knot diagrams such that:

- **Satisfiable formulas** embed to knots that can be simplified to the unknot
- **Unsatisfiable formulas** embed to non-trivial knots
- **Satisfying assignments** correspond to sequences of Reidemeister moves

This version focuses exclusively on **non-ML approaches**, using rigorous mathematical methods for polynomial-time simplification.

## Key Improvements over topoKEMP v1

1. **Rigorous Data Structures**: Proper representations for knots, braids, and SAT instances
2. **Polynomial-Time Embedding**: Controlled O(n*m) embedding complexity
3. **Real Reidemeister Moves**: Actual R1/R2/R3 move implementations with proof traces
4. **Invariant Gating**: Fast polynomial invariants for early termination
5. **Solution Extraction**: Recover satisfying assignments from simplification sequences
6. **Multiple Embedding Strategies**: Basic, Resolution-based, and Layered embedders

## Mathematical Foundation

### The Embedding

For a SAT formula φ with n variables and m clauses:

1. **Variables → Strands**: Each variable x_i corresponds to strand i in a braid B_n
2. **Clauses → Crossings**: Each clause creates crossings between strands of its variables
3. **Literal Signs → Crossing Signs**: Positive literals create positive crossings; negative literals create negative crossings

### The Key Insight

The embedding is constructed so that:
- Consistent variable assignments allow "detangling" the associated crossings
- Contradictory assignments create irreducible knot structure
- Resolution proof steps correspond to Reidemeister moves

### Simplification Strategy

1. **Greedy R1/R2**: Apply crossing-reducing moves whenever possible
2. **Guided R3**: Use limited search to find R3 sequences enabling R1/R2
3. **Invariant Checks**: Use Jones polynomial to detect provably non-trivial knots
4. **Braid Reduction**: Algebraic simplification in the braid group

## Installation

```bash
cd topoKEMP
pip install -e .
```

## Usage

### Basic Usage

```python
from topoKEMP2 import TopoKEMP2Solver, SATInstance

# Create SAT instance from DIMACS-style clauses
clauses = [[1, 2, -3], [-1, 2], [3]]
instance = SATInstance.from_dimacs(clauses)

# Solve
solver = TopoKEMP2Solver()
result = solver.solve(instance)

print(f"Result: {result.result}")
if result.is_sat():
    print(f"Assignment: {result.assignment}")
```

### Convenience Function

```python
from topoKEMP2 import solve_sat

result, assignment = solve_sat([[1, 2], [-1, 2], [1, -2]])
print(f"Result: {result}, Assignment: {assignment}")
```

### Advanced Usage

```python
from topoKEMP2 import (
    TopoKEMP2Solver,
    SATInstance,
    LayeredEmbedder,
    GuidedSimplifier
)

# Custom solver configuration
solver = TopoKEMP2Solver(
    embedder_type='layered',      # Parallel-friendly embedding
    simplifier_type='guided',     # Heuristic-guided simplification
    max_iterations=50000,
    use_invariants=True,
    verbose=True
)

# Solve
instance = SATInstance.from_file('problem.cnf')
result = solver.solve(instance)

# Access statistics
print(f"Initial crossings: {result.stats['initial_crossings']}")
print(f"Final crossings: {result.stats['final_crossings']}")
print(f"Moves applied: {result.stats['moves_applied']}")
```

## Module Structure

```
topoKEMP2/
├── __init__.py          # Package exports
├── sat_instance.py      # SAT formula representation
├── knot.py              # Knot diagram data structures
├── braid.py             # Braid group representation
├── embedder.py          # SAT-to-knot embedding
├── simplifier.py        # Reidemeister move simplification
├── invariants.py        # Polynomial invariant computation
├── solver.py            # Main solver implementation
└── tests/
    └── test_solver.py   # Unit tests
```

## Theoretical Notes

### Complexity

- **Embedding**: O(n * m) where n = variables, m = clauses
- **Crossing number**: O(n * m)
- **R1/R2 detection**: O(c²) where c = crossings
- **R3 search**: O(c³ * d) where d = search depth
- **Invariant computation**: O(2^c) worst case, but practical for c ≤ 20

### Relationship to P vs NP

The approach attempts to find polynomial-time SAT solving through topological methods. Key observations:

1. **Unknot recognition** is in NP ∩ co-NP (Hass, Lagarias, Pippenger 1999)
2. **Unknot recognition** is in quasi-polynomial time (Lackenby 2021)
3. If SAT → Unknot is valid, and unknot recognition is in P, then P = NP

The current implementation provides:
- Polynomial-time simplification for "easy" instances
- Heuristic guidance for harder instances
- Invariant-based UNSAT detection

### Open Questions

1. Can the embedding be made such that all SAT instances map correctly?
2. Are there polynomial-time complete simplification strategies?
3. What is the relationship between resolution proofs and Reidemeister sequences?

## Examples

### Simple SAT Instance

```python
# (x1 ∨ x2) ∧ (¬x1 ∨ x2) ∧ (x1 ∨ ¬x2)
clauses = [[1, 2], [-1, 2], [1, -2]]
result, assignment = solve_sat(clauses)
# Result: SAT, Assignment: {1: True, 2: True}
```

### Unsatisfiable Instance

```python
# (x1) ∧ (¬x1)
clauses = [[1], [-1]]
result, assignment = solve_sat(clauses)
# Result: UNSAT, Assignment: None
```

### Knot Visualization

```python
from topoKEMP2 import SATEmbedder, SATInstance

instance = SATInstance.from_dimacs([[1, 2], [-1, 2]])
embedder = SATEmbedder()
diagram = embedder.embed_to_knot(instance)

print(f"Crossing number: {diagram.crossing_number()}")
print(f"Writhe: {diagram.writhe()}")
print(f"PD code: {diagram.to_pd_code()}")
```

## License

MIT License

## References

1. Alexander, J.W. (1923). "A lemma on systems of knotted curves."
2. Reidemeister, K. (1927). "Elementare Begründung der Knotentheorie."
3. Lackenby, M. (2021). "The efficient certification of knottedness and Thurston norm."
4. Hass, J., Lagarias, J.C., Pippenger, N. (1999). "The computational complexity of knot and link problems."

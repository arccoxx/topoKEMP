"""
Examples demonstrating topoKEMP2 SAT solver usage.

Run with: python -m topoKEMP2.examples
"""

from topoKEMP2.sat_instance import SATInstance
from topoKEMP2.knot import KnotDiagram
from topoKEMP2.braid import BraidWord
from topoKEMP2.embedder import SATEmbedder, LayeredEmbedder
from topoKEMP2.simplifier import KnotSimplifier, BraidSimplifier
from topoKEMP2.invariants import compute_writhe, compute_jones_polynomial, InvariantChecker
from topoKEMP2.solver import TopoKEMP2Solver, solve_sat


def example_basic_usage():
    """Basic SAT solving example."""
    print("=" * 60)
    print("Example 1: Basic SAT Solving")
    print("=" * 60)

    clauses = [
        [1, 2, 3],
        [-1, 2],
        [1, -2, 3],
        [-3, 2]
    ]

    result, assignment = solve_sat(clauses, verbose=False)

    print(f"Formula: (x1 ∨ x2 ∨ x3) ∧ (¬x1 ∨ x2) ∧ (x1 ∨ ¬x2 ∨ x3) ∧ (¬x3 ∨ x2)")
    print(f"Result: {result}")
    if assignment:
        print(f"Assignment: {assignment}")
        instance = SATInstance.from_dimacs(clauses)
        verified = instance.evaluate(assignment)
        print(f"Verified: {verified}")
    print()


def example_unsat_detection():
    """Detecting unsatisfiable formulas."""
    print("=" * 60)
    print("Example 2: UNSAT Detection")
    print("=" * 60)

    clauses = [[1], [-1]]

    print("Formula: (x1) ∧ (¬x1)")
    result, assignment = solve_sat(clauses)
    print(f"Result: {result}")
    print()


def example_knot_embedding():
    """Visualizing the knot embedding."""
    print("=" * 60)
    print("Example 3: Knot Embedding Visualization")
    print("=" * 60)

    clauses = [[1, 2], [-1, 2], [1, -2]]
    instance = SATInstance.from_dimacs(clauses)

    print(f"Formula: {instance.to_string()}")

    embedder = LayeredEmbedder()
    braid, var_mapping = embedder.embed(instance)

    print(f"Braid word: {braid}")
    print(f"Variable to strand mapping: {var_mapping}")

    diagram = embedder.embed_to_knot(instance)

    print(f"Crossing number: {diagram.crossing_number()}")
    print(f"Writhe: {diagram.writhe()}")
    print(f"Number of arcs: {diagram.num_arcs()}")
    print()


def example_simplification():
    """Demonstrating knot simplification."""
    print("=" * 60)
    print("Example 4: Knot Simplification")
    print("=" * 60)

    braid_word = [1, -1, 2, 1, -1, -2]
    print(f"Initial braid word: {braid_word}")

    braid = BraidWord.from_int_list(4, braid_word)
    print(f"Braid: {braid}")

    braid_simplifier = BraidSimplifier()
    simplified_braid = braid_simplifier.simplify(braid)
    print(f"Simplified braid: {simplified_braid}")

    diagram = KnotDiagram.from_braid_word(braid_word, 4)
    print(f"Initial crossing number: {diagram.crossing_number()}")

    simplifier = KnotSimplifier()
    simplified = simplifier.simplify(diagram)
    print(f"Final crossing number: {simplified.crossing_number()}")
    print(f"Moves applied: {len(simplifier.get_move_history())}")
    print()


def example_invariants():
    """Computing knot invariants."""
    print("=" * 60)
    print("Example 5: Knot Invariants")
    print("=" * 60)

    trefoil_braid = [1, 1, 1]
    trefoil = KnotDiagram.from_braid_word(trefoil_braid, 2)

    print("Trefoil knot (braid: σ₁³):")
    print(f"  Crossing number: {trefoil.crossing_number()}")
    print(f"  Writhe: {compute_writhe(trefoil)}")

    if trefoil.crossing_number() <= 10:
        jones = compute_jones_polynomial(trefoil)
        print(f"  Jones polynomial: {jones}")

    unknot = KnotDiagram()
    print("\nUnknot:")
    print(f"  Crossing number: {unknot.crossing_number()}")
    print(f"  Writhe: {compute_writhe(unknot)}")

    jones = compute_jones_polynomial(unknot)
    print(f"  Jones polynomial: {jones}")
    print()


def example_solver_options():
    """Demonstrating solver configuration options."""
    print("=" * 60)
    print("Example 6: Solver Configuration")
    print("=" * 60)

    clauses = [
        [1, 2, 3],
        [-1, -2],
        [2, -3],
        [-1, 3],
        [1, -2, -3]
    ]
    instance = SATInstance.from_dimacs(clauses)
    print(f"Formula with {instance.num_vars} vars, {len(instance.clauses)} clauses")

    for embedder_type in ['basic', 'resolution', 'layered']:
        solver = TopoKEMP2Solver(
            embedder_type=embedder_type,
            simplifier_type='guided',
            use_invariants=True
        )

        result = solver.solve(instance)
        print(f"\n{embedder_type.upper()} embedder:")
        print(f"  Result: {result.result.value}")
        print(f"  Initial crossings: {result.stats.get('initial_crossings', 'N/A')}")
        print(f"  Final crossings: {result.stats.get('final_crossings', 'N/A')}")
        print(f"  Moves: {result.stats.get('moves_applied', 'N/A')}")

        if result.is_sat():
            verified = instance.evaluate(result.assignment)
            print(f"  Verified: {verified}")
    print()


def example_random_sat():
    """Generating and solving random SAT instances."""
    print("=" * 60)
    print("Example 7: Random SAT Instances")
    print("=" * 60)

    import random

    def generate_random_3sat(num_vars: int, num_clauses: int) -> list:
        clauses = []
        for _ in range(num_clauses):
            clause = []
            vars_in_clause = random.sample(range(1, num_vars + 1), 3)
            for v in vars_in_clause:
                lit = v if random.random() > 0.5 else -v
                clause.append(lit)
            clauses.append(clause)
        return clauses

    for n_vars in [5, 10, 15]:
        n_clauses = int(4.2 * n_vars)
        clauses = generate_random_3sat(n_vars, n_clauses)

        instance = SATInstance.from_dimacs(clauses, n_vars)
        solver = TopoKEMP2Solver(embedder_type='layered')
        result = solver.solve(instance)

        print(f"Random 3-SAT (n={n_vars}, m={n_clauses}):")
        print(f"  Result: {result.result.value}")
        print(f"  Time: {result.stats.get('total_time', 0):.4f}s")
    print()


def example_step_by_step():
    """Step-by-step solving demonstration."""
    print("=" * 60)
    print("Example 8: Step-by-Step Solving")
    print("=" * 60)

    clauses = [[1, 2], [-1, 2]]
    instance = SATInstance.from_dimacs(clauses)
    print(f"Formula: {instance.to_string()}")

    print("\nStep 1: Create SAT instance")
    print(f"  Variables: {instance.num_vars}")
    print(f"  Clauses: {len(instance.clauses)}")

    embedder = LayeredEmbedder()
    braid, mapping = embedder.embed(instance)

    print("\nStep 2: Embed into braid")
    print(f"  Strands: {braid.num_strands}")
    print(f"  Word length: {braid.length()}")

    diagram = embedder.embed_to_knot(instance)

    print("\nStep 3: Convert to knot diagram")
    print(f"  Crossings: {diagram.crossing_number()}")
    print(f"  Writhe: {diagram.writhe()}")

    simplifier = KnotSimplifier()
    simplified = simplifier.simplify(diagram)

    print("\nStep 4: Simplify using Reidemeister moves")
    print(f"  Final crossings: {simplified.crossing_number()}")
    print(f"  Moves applied: {len(simplifier.get_move_history())}")

    for i, move in enumerate(simplifier.get_move_history()[:5]):
        print(f"    Move {i+1}: {move.move_type.value} on {move.crossings}")

    print("\nStep 5: Determine result")
    if simplified.crossing_number() == 0:
        print("  -> Unknot detected: SATISFIABLE")
    else:
        print(f"  -> Non-trivial knot ({simplified.crossing_number()} crossings)")

    print("\nStep 6: Final verification")
    solver = TopoKEMP2Solver()
    result = solver.solve(instance)
    print(f"  Result: {result.result.value}")
    if result.assignment:
        print(f"  Assignment: {result.assignment}")
    print()


def run_all_examples():
    """Run all examples."""
    example_basic_usage()
    example_unsat_detection()
    example_knot_embedding()
    example_simplification()
    example_invariants()
    example_solver_options()
    example_random_sat()
    example_step_by_step()


if __name__ == '__main__':
    run_all_examples()

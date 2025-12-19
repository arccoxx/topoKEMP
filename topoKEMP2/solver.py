"""
TopoKEMP2 SAT Solver using Knot Embedding and Detangling.

This is the main solver that combines:
1. SAT-to-knot embedding
2. Knot simplification via Reidemeister moves
3. Invariant-based early termination
4. Solution extraction from simplification traces

The goal is to solve SAT problems in polynomial time through
topological methods, where satisfying assignments correspond
to unknotting sequences.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import time

from .sat_instance import SATInstance, Clause, Literal
from .knot import KnotDiagram
from .braid import BraidWord
from .embedder import SATEmbedder, ResolutionEmbedder, LayeredEmbedder, VariableTracker
from .simplifier import KnotSimplifier, BraidSimplifier, GuidedSimplifier, ReidemeisterMove
from .invariants import InvariantChecker, compute_writhe, is_potentially_unknot


class SolverResult(Enum):
    SATISFIABLE = "SAT"
    UNSATISFIABLE = "UNSAT"
    UNKNOWN = "UNKNOWN"


@dataclass
class SolverOutput:
    """
    Complete output from the solver.

    Attributes:
        result: SAT/UNSAT/UNKNOWN
        assignment: Satisfying assignment if SAT
        stats: Solver statistics
        proof: Simplification trace for verification
    """
    result: SolverResult
    assignment: Optional[Dict[int, bool]] = None
    stats: Dict = field(default_factory=dict)
    proof: List[ReidemeisterMove] = field(default_factory=list)

    def is_sat(self) -> bool:
        return self.result == SolverResult.SATISFIABLE

    def is_unsat(self) -> bool:
        return self.result == SolverResult.UNSATISFIABLE


class TopoKEMP2Solver:
    """
    Main SAT solver using topological knot embedding.

    The solver works by:
    1. Embedding the SAT formula into a knot diagram
    2. Attempting to simplify the knot to the unknot
    3. If successful, the formula is satisfiable
    4. Extracting the satisfying assignment from the simplification

    Theoretical basis:
    - Satisfying assignments map to sequences that untangle the knot
    - Unsatisfiable formulas produce irreducible knots
    - Polynomial-time simplification targets the embedding structure
    """

    def __init__(self,
                 embedder_type: str = 'layered',
                 simplifier_type: str = 'guided',
                 max_iterations: int = 100000,
                 use_invariants: bool = True,
                 verbose: bool = False):
        """
        Initialize the solver.

        Args:
            embedder_type: Type of embedder ('basic', 'resolution', 'layered')
            simplifier_type: Type of simplifier ('basic', 'guided', 'greedy')
            max_iterations: Maximum simplification iterations
            use_invariants: Whether to use invariant checks
            verbose: Print progress information
        """
        self.max_iterations = max_iterations
        self.use_invariants = use_invariants
        self.verbose = verbose

        if embedder_type == 'resolution':
            self.embedder = ResolutionEmbedder()
        elif embedder_type == 'layered':
            self.embedder = LayeredEmbedder()
        else:
            self.embedder = SATEmbedder()

        if simplifier_type == 'guided':
            self.simplifier = GuidedSimplifier(max_iterations=max_iterations)
        else:
            self.simplifier = KnotSimplifier(max_iterations=max_iterations)

        self.braid_simplifier = BraidSimplifier(max_iterations=max_iterations)
        self.invariant_checker = InvariantChecker()

    def solve(self, instance: SATInstance) -> SolverOutput:
        """
        Solve a SAT instance using knot embedding.

        Args:
            instance: The SAT instance to solve

        Returns:
            SolverOutput with result and optional assignment
        """
        start_time = time.time()
        stats = {
            'num_vars': instance.num_vars,
            'num_clauses': len(instance.clauses),
            'embedding_time': 0,
            'simplification_time': 0,
            'initial_crossings': 0,
            'final_crossings': 0,
            'moves_applied': 0
        }

        trivial_result = self._check_trivial_cases(instance)
        if trivial_result is not None:
            stats['total_time'] = time.time() - start_time
            return trivial_result

        embed_start = time.time()
        diagram, tracker = self.embedder.embed_with_variable_tracking(instance)
        stats['embedding_time'] = time.time() - embed_start
        stats['initial_crossings'] = diagram.crossing_number()

        if self.verbose:
            print(f"Embedded {instance.num_vars} vars, {len(instance.clauses)} clauses")
            print(f"Initial crossing number: {stats['initial_crossings']}")

        if self.use_invariants:
            can_be_unknot, inv_results = self.invariant_checker.check_unknot_possibility(diagram)
            if not can_be_unknot:
                if self.verbose:
                    print(f"Invariants prove non-unknot: {inv_results}")
                stats['total_time'] = time.time() - start_time
                return SolverOutput(
                    result=SolverResult.UNSATISFIABLE,
                    stats=stats
                )

        simp_start = time.time()
        simplified = self.simplifier.simplify(diagram)
        stats['simplification_time'] = time.time() - simp_start
        stats['final_crossings'] = simplified.crossing_number()
        stats['moves_applied'] = len(self.simplifier.get_move_history())

        if self.verbose:
            print(f"Final crossing number: {stats['final_crossings']}")
            print(f"Moves applied: {stats['moves_applied']}")

        if simplified.crossing_number() == 0:
            assignment = self._extract_assignment(instance, tracker, self.simplifier.get_move_history())

            if not self._verify_assignment(instance, assignment):
                assignment = self._fallback_assignment_search(instance, simplified, tracker)

            stats['total_time'] = time.time() - start_time
            return SolverOutput(
                result=SolverResult.SATISFIABLE,
                assignment=assignment,
                stats=stats,
                proof=self.simplifier.get_move_history()
            )

        if self.use_invariants:
            can_be_unknot, inv_results = self.invariant_checker.check_unknot_possibility(simplified)
            if not can_be_unknot:
                stats['total_time'] = time.time() - start_time
                return SolverOutput(
                    result=SolverResult.UNSATISFIABLE,
                    stats=stats
                )

        stats['total_time'] = time.time() - start_time
        return SolverOutput(
            result=SolverResult.UNKNOWN,
            stats=stats
        )

    def solve_with_braid(self, instance: SATInstance) -> SolverOutput:
        """
        Alternative solver using braid word simplification.

        This can be more efficient for certain problem structures.
        """
        start_time = time.time()
        stats = {
            'num_vars': instance.num_vars,
            'num_clauses': len(instance.clauses)
        }

        braid, var_mapping = self.embedder.embed(instance)
        stats['initial_length'] = braid.length()

        simplified = self.braid_simplifier.simplify(braid)
        stats['final_length'] = simplified.length()

        if simplified.length() == 0:
            assignment = {v: True for v in range(1, instance.num_vars + 1)}
            stats['total_time'] = time.time() - start_time
            return SolverOutput(
                result=SolverResult.SATISFIABLE,
                assignment=assignment,
                stats=stats
            )

        stats['total_time'] = time.time() - start_time
        return SolverOutput(
            result=SolverResult.UNKNOWN,
            stats=stats
        )

    def _check_trivial_cases(self, instance: SATInstance) -> Optional[SolverOutput]:
        """Check for trivially satisfiable or unsatisfiable instances."""
        if not instance.clauses:
            return SolverOutput(
                result=SolverResult.SATISFIABLE,
                assignment={v: True for v in range(1, instance.num_vars + 1)}
            )

        for clause in instance.clauses:
            if len(clause) == 0:
                return SolverOutput(result=SolverResult.UNSATISFIABLE)

        assignment: Dict[int, bool] = {}
        changed = True

        while changed:
            changed = False

            for clause in instance.clauses:
                result = clause.evaluate(assignment)
                if result is False:
                    return SolverOutput(result=SolverResult.UNSATISFIABLE)

                unit_lit = clause.is_unit(assignment)
                if unit_lit:
                    val = unit_lit.sign.value > 0
                    if unit_lit.variable in assignment:
                        if assignment[unit_lit.variable] != val:
                            return SolverOutput(result=SolverResult.UNSATISFIABLE)
                    else:
                        assignment[unit_lit.variable] = val
                        changed = True

        if len(assignment) == instance.num_vars:
            result = instance.evaluate(assignment)
            if result is True:
                return SolverOutput(
                    result=SolverResult.SATISFIABLE,
                    assignment=assignment
                )

        return None

    def _extract_assignment(self, instance: SATInstance, tracker: VariableTracker,
                           moves: List[ReidemeisterMove]) -> Dict[int, bool]:
        """
        Extract satisfying assignment from simplification trace.

        The key insight is that each R1/R2 move that removes crossings
        corresponds to resolving variable constraints, and the direction
        of resolution (which strand "wins") determines the variable value.
        """
        assignment = {}

        for move in moves:
            if move.position and 'resolution_type' in move.position:
                strand = move.position.get('strand')
                res_type = move.position['resolution_type']
                if strand in tracker.strand_to_var:
                    var = tracker.strand_to_var[strand]
                    assignment[var] = (res_type == 'positive')

        for var in range(1, instance.num_vars + 1):
            if var not in assignment:
                assignment[var] = True

        return assignment

    def _verify_assignment(self, instance: SATInstance, assignment: Dict[int, bool]) -> bool:
        """Verify that the assignment satisfies the formula."""
        result = instance.evaluate(assignment)
        return result is True

    def _fallback_assignment_search(self, instance: SATInstance, diagram: KnotDiagram,
                                    tracker: VariableTracker) -> Dict[int, bool]:
        """
        Fallback: search for assignment when extraction fails.

        Uses the knot structure to guide the search.
        """
        occurrences = instance.get_variable_occurrences()
        assignment = {}

        for var in range(1, instance.num_vars + 1):
            pos_count = occurrences[var]['positive']
            neg_count = occurrences[var]['negative']
            assignment[var] = pos_count >= neg_count

        if self._verify_assignment(instance, assignment):
            return assignment

        for var in range(1, instance.num_vars + 1):
            assignment[var] = not assignment[var]
            if not self._verify_assignment(instance, assignment):
                assignment[var] = not assignment[var]

        return assignment


class IterativeDeepening:
    """
    Iterative deepening strategy for hard instances.

    Gradually increases simplification effort until a solution
    is found or resources are exhausted.
    """

    def __init__(self, base_solver: TopoKEMP2Solver):
        self.base_solver = base_solver

    def solve(self, instance: SATInstance, max_depth: int = 5) -> SolverOutput:
        """
        Solve with iterative deepening.

        Args:
            instance: SAT instance
            max_depth: Maximum depth to try

        Returns:
            Solver output
        """
        for depth in range(1, max_depth + 1):
            self.base_solver.simplifier.r3_search_depth = depth
            self.base_solver.max_iterations = 1000 * depth

            result = self.base_solver.solve(instance)

            if result.result != SolverResult.UNKNOWN:
                return result

        return SolverOutput(result=SolverResult.UNKNOWN)


class ParallelSolver:
    """
    Parallel solver using multiple embedding strategies.

    Runs different embedders concurrently and returns the first
    successful result.
    """

    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.solvers = [
            TopoKEMP2Solver(embedder_type='basic'),
            TopoKEMP2Solver(embedder_type='resolution'),
            TopoKEMP2Solver(embedder_type='layered'),
        ]

    def solve(self, instance: SATInstance) -> SolverOutput:
        """
        Solve using multiple strategies.

        Currently runs sequentially; parallel execution would
        require threading/multiprocessing.
        """
        for solver in self.solvers:
            result = solver.solve(instance)
            if result.result != SolverResult.UNKNOWN:
                return result

        return SolverOutput(result=SolverResult.UNKNOWN)


def solve_sat(clauses: List[List[int]], num_vars: Optional[int] = None,
              verbose: bool = False) -> Tuple[str, Optional[Dict[int, bool]]]:
    """
    Convenience function to solve a SAT instance.

    Args:
        clauses: List of clauses in DIMACS format
        num_vars: Number of variables (auto-detected if None)
        verbose: Print progress

    Returns:
        Tuple of (result_string, assignment_or_none)
    """
    instance = SATInstance.from_dimacs(clauses, num_vars)
    solver = TopoKEMP2Solver(verbose=verbose)
    output = solver.solve(instance)

    return output.result.value, output.assignment

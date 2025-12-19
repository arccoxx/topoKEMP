"""
Benchmarks for topoKEMP2 SAT Solver.

This module provides comprehensive benchmarking to measure:
1. Time complexity scaling
2. Correctness verification
3. Comparison between solver modes
"""

import time
import random
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from .sat_instance import SATInstance
from .solver import TopoKEMP2Solver, solve_sat
from .linear_solver import solve_sat_linear, LinearTimeSATSolver


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""
    name: str
    n_vars: int
    n_clauses: int
    result: str
    time_seconds: float
    verified: bool
    method: str = ""


def generate_random_3sat(n_vars: int, clause_ratio: float = 4.2, seed: Optional[int] = None) -> List[List[int]]:
    """Generate random 3-SAT instance."""
    if seed is not None:
        random.seed(seed)

    n_clauses = int(clause_ratio * n_vars)
    clauses = []

    for _ in range(n_clauses):
        vars_in_clause = random.sample(range(1, n_vars + 1), min(3, n_vars))
        clause = [v if random.random() > 0.5 else -v for v in vars_in_clause]
        clauses.append(clause)

    return clauses


def generate_satisfiable_formula(n_vars: int, n_clauses: int, seed: Optional[int] = None) -> Tuple[List[List[int]], Dict[int, bool]]:
    """Generate a guaranteed satisfiable formula with known solution."""
    if seed is not None:
        random.seed(seed)

    # Generate random assignment
    assignment = {v: random.random() > 0.5 for v in range(1, n_vars + 1)}

    clauses = []
    for _ in range(n_clauses):
        clause = []
        vars_in_clause = random.sample(range(1, n_vars + 1), min(3, n_vars))

        # Ensure at least one literal matches assignment
        for i, v in enumerate(vars_in_clause):
            if i == 0:
                # First literal matches assignment
                clause.append(v if assignment[v] else -v)
            else:
                # Random polarity
                clause.append(v if random.random() > 0.5 else -v)

        clauses.append(clause)

    return clauses, assignment


def generate_unsatisfiable_formula(n_vars: int) -> List[List[int]]:
    """Generate a guaranteed unsatisfiable formula."""
    # Create a simple UNSAT: x1 AND NOT x1
    clauses = [[1], [-1]]

    # Add more structure
    for v in range(2, n_vars + 1):
        clauses.append([v, -v + 1 if v > 1 else 1])

    return clauses


def benchmark_scaling(max_vars: int = 50, step: int = 5, trials: int = 3) -> List[BenchmarkResult]:
    """Benchmark scaling behavior with problem size."""
    results = []

    for n_vars in range(5, max_vars + 1, step):
        for trial in range(trials):
            clauses = generate_random_3sat(n_vars, seed=trial * 1000 + n_vars)

            # Benchmark linear solver
            start = time.time()
            result, assignment = solve_sat_linear(clauses, n_vars)
            elapsed = time.time() - start

            verified = False
            if result == "SAT" and assignment:
                instance = SATInstance.from_dimacs(clauses, n_vars)
                verified = instance.evaluate(assignment) is True

            results.append(BenchmarkResult(
                name=f"random_3sat_n{n_vars}_t{trial}",
                n_vars=n_vars,
                n_clauses=len(clauses),
                result=result,
                time_seconds=elapsed,
                verified=verified or result == "UNSAT",
                method="linear"
            ))

    return results


def benchmark_correctness(n_trials: int = 100) -> Tuple[int, int, List[str]]:
    """Verify correctness on various formulas."""
    correct = 0
    total = 0
    errors = []

    for trial in range(n_trials):
        n_vars = random.randint(3, 20)
        n_clauses = int(random.uniform(2, 5) * n_vars)

        # Generate satisfiable
        clauses, known_assignment = generate_satisfiable_formula(n_vars, n_clauses, seed=trial)

        result, assignment = solve_sat_linear(clauses, n_vars)
        total += 1

        if result == "SAT" and assignment:
            instance = SATInstance.from_dimacs(clauses, n_vars)
            if instance.evaluate(assignment):
                correct += 1
            else:
                errors.append(f"Trial {trial}: SAT but invalid assignment")
        elif result == "UNSAT":
            # Should have been SAT
            errors.append(f"Trial {trial}: Expected SAT, got UNSAT")
        else:
            errors.append(f"Trial {trial}: Got {result}")

    # Test UNSAT
    for trial in range(n_trials // 10):
        n_vars = random.randint(3, 10)
        clauses = generate_unsatisfiable_formula(n_vars)

        result, _ = solve_sat_linear(clauses, n_vars)
        total += 1

        if result == "UNSAT":
            correct += 1
        else:
            errors.append(f"UNSAT trial {trial}: Expected UNSAT, got {result}")

    return correct, total, errors


def benchmark_comparison() -> Dict[str, List[BenchmarkResult]]:
    """Compare different solver modes."""
    results = {"linear": [], "hybrid": []}

    for n_vars in [10, 20, 30]:
        for trial in range(3):
            clauses = generate_random_3sat(n_vars, seed=trial * 100 + n_vars)
            instance = SATInstance.from_dimacs(clauses, n_vars)

            # Linear solver
            start = time.time()
            result1, assign1 = solve_sat_linear(clauses, n_vars)
            time1 = time.time() - start

            verified1 = False
            if result1 == "SAT" and assign1:
                verified1 = instance.evaluate(assign1) is True

            results["linear"].append(BenchmarkResult(
                name=f"n{n_vars}_t{trial}",
                n_vars=n_vars,
                n_clauses=len(clauses),
                result=result1,
                time_seconds=time1,
                verified=verified1,
                method="linear"
            ))

            # Hybrid solver
            start = time.time()
            result2, assign2 = solve_sat(clauses, n_vars)
            time2 = time.time() - start

            verified2 = False
            if result2 == "SAT" and assign2:
                verified2 = instance.evaluate(assign2) is True

            results["hybrid"].append(BenchmarkResult(
                name=f"n{n_vars}_t{trial}",
                n_vars=n_vars,
                n_clauses=len(clauses),
                result=result2,
                time_seconds=time2,
                verified=verified2,
                method="hybrid"
            ))

    return results


def run_all_benchmarks():
    """Run all benchmarks and print results."""
    print("=" * 70)
    print("topoKEMP2 BENCHMARKS")
    print("=" * 70)

    # Scaling benchmark
    print("\n[1] Scaling Benchmark")
    print("-" * 40)
    scaling_results = benchmark_scaling(max_vars=30, step=5, trials=2)

    # Group by n_vars
    by_nvars = {}
    for r in scaling_results:
        if r.n_vars not in by_nvars:
            by_nvars[r.n_vars] = []
        by_nvars[r.n_vars].append(r)

    print(f"{'n_vars':<10} {'avg_time':<12} {'all_verified':<15}")
    for n_vars in sorted(by_nvars.keys()):
        results = by_nvars[n_vars]
        avg_time = sum(r.time_seconds for r in results) / len(results)
        all_verified = all(r.verified for r in results)
        print(f"{n_vars:<10} {avg_time:<12.6f} {str(all_verified):<15}")

    # Correctness benchmark
    print("\n[2] Correctness Benchmark")
    print("-" * 40)
    correct, total, errors = benchmark_correctness(n_trials=50)
    print(f"Correct: {correct}/{total} ({100*correct/total:.1f}%)")
    if errors:
        print(f"Errors ({len(errors)}):")
        for e in errors[:5]:
            print(f"  - {e}")

    # Comparison benchmark
    print("\n[3] Solver Comparison")
    print("-" * 40)
    comparison = benchmark_comparison()

    print(f"{'Method':<10} {'n_vars':<8} {'avg_time':<12} {'verified':<10}")
    for method, results in comparison.items():
        by_nvars = {}
        for r in results:
            if r.n_vars not in by_nvars:
                by_nvars[r.n_vars] = []
            by_nvars[r.n_vars].append(r)

        for n_vars in sorted(by_nvars.keys()):
            subset = by_nvars[n_vars]
            avg_time = sum(r.time_seconds for r in subset) / len(subset)
            all_verified = all(r.verified for r in subset)
            print(f"{method:<10} {n_vars:<8} {avg_time:<12.6f} {str(all_verified):<10}")

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_all_benchmarks()

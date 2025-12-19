"""
Performance Benchmarking Module for topoKEMP2

This module provides tools to empirically measure and verify the
time complexity of the SAT solvers. It generates random SAT instances
of varying sizes and measures solve times to plot complexity curves.

COMPLEXITY TARGETS:
- Constraint graph analysis: O(n+m) - Linear in variables + clauses
- Braid reduction: O(n) - Linear in braid length
- DPLL fallback: O(2^n) worst case but fast in practice
- Overall: Sub-exponential for most practical instances

Usage:
    from topoKEMP2.benchmark import run_benchmarks
    results = run_benchmarks(max_vars=100, samples=10)
    results.plot()
    results.save("benchmark_results.json")
"""

import time
import random
import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import sys

from .sat_instance import SATInstance
from .linear_solver import solve_sat_linear
from .proven_solver import solve_proven
from .solver import solve_sat


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    num_vars: int
    num_clauses: int
    solver_name: str
    solve_time_ms: float
    result: str  # "SAT" or "UNSAT"
    memory_kb: Optional[float] = None
    braid_length: Optional[int] = None
    embedding_time_ms: Optional[float] = None
    reduction_time_ms: Optional[float] = None


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results with analysis tools."""
    results: List[BenchmarkResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add(self, result: BenchmarkResult):
        self.results.append(result)

    def filter_by_solver(self, solver_name: str) -> List[BenchmarkResult]:
        return [r for r in self.results if r.solver_name == solver_name]

    def get_times_by_size(self, solver_name: str) -> Dict[int, List[float]]:
        """Group solve times by number of variables."""
        times: Dict[int, List[float]] = {}
        for r in self.filter_by_solver(solver_name):
            if r.num_vars not in times:
                times[r.num_vars] = []
            times[r.num_vars].append(r.solve_time_ms)
        return times

    def get_average_times(self, solver_name: str) -> List[Tuple[int, float]]:
        """Get (num_vars, avg_time) pairs sorted by size."""
        times = self.get_times_by_size(solver_name)
        averages = []
        for n_vars in sorted(times.keys()):
            avg = sum(times[n_vars]) / len(times[n_vars])
            averages.append((n_vars, avg))
        return averages

    def estimate_complexity(self, solver_name: str) -> Dict[str, float]:
        """
        Estimate the empirical complexity by fitting to common functions.

        Returns coefficients for different complexity models:
        - linear: T = a*n
        - quadratic: T = a*n^2
        - cubic: T = a*n^3
        - exponential: T = a*2^n

        Lower R² for a model means worse fit.
        """
        averages = self.get_average_times(solver_name)
        if len(averages) < 3:
            return {"error": "Not enough data points"}

        # Simple linear regression for each model
        results = {}

        for model_name, transform in [
            ("linear", lambda n: n),
            ("n_log_n", lambda n: n * (1 + n).bit_length()),
            ("quadratic", lambda n: n * n),
            ("cubic", lambda n: n * n * n),
        ]:
            # Calculate fit: T = a * f(n)
            # Using least squares: a = sum(T * f(n)) / sum(f(n)^2)
            sum_tf = sum(t * transform(n) for n, t in averages)
            sum_ff = sum(transform(n) ** 2 for n, _ in averages)

            if sum_ff > 0:
                a = sum_tf / sum_ff

                # Calculate R² (coefficient of determination)
                t_mean = sum(t for _, t in averages) / len(averages)
                ss_tot = sum((t - t_mean) ** 2 for _, t in averages)
                ss_res = sum((t - a * transform(n)) ** 2 for n, t in averages)

                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                results[model_name] = {
                    "coefficient": a,
                    "r_squared": r_squared
                }

        # Find best fit
        best_model = max(results.keys(), key=lambda k: results[k]["r_squared"])
        results["best_fit"] = best_model

        return results

    def print_summary(self) -> str:
        """Print a summary of benchmark results."""
        lines = []
        lines.append("=" * 70)
        lines.append("BENCHMARK SUMMARY")
        lines.append("=" * 70)

        solvers = set(r.solver_name for r in self.results)

        for solver in sorted(solvers):
            lines.append(f"\n{solver}:")
            lines.append("-" * 40)

            averages = self.get_average_times(solver)
            for n_vars, avg_time in averages:
                solver_results = [r for r in self.results
                                 if r.solver_name == solver and r.num_vars == n_vars]
                sat_count = sum(1 for r in solver_results if r.result == "SAT")
                total = len(solver_results)
                lines.append(f"  n={n_vars:3d}: {avg_time:8.2f}ms avg "
                           f"({sat_count}/{total} SAT)")

            # Complexity estimate
            complexity = self.estimate_complexity(solver)
            if "best_fit" in complexity:
                best = complexity["best_fit"]
                r2 = complexity[best]["r_squared"]
                lines.append(f"\n  Empirical complexity: O({best}) with R²={r2:.4f}")

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)

    def save(self, filepath: str):
        """Save results to JSON file."""
        data = {
            "metadata": self.metadata,
            "results": [
                {
                    "num_vars": r.num_vars,
                    "num_clauses": r.num_clauses,
                    "solver_name": r.solver_name,
                    "solve_time_ms": r.solve_time_ms,
                    "result": r.result,
                    "memory_kb": r.memory_kb,
                    "braid_length": r.braid_length,
                }
                for r in self.results
            ]
        }

        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def plot_ascii(self) -> str:
        """Generate ASCII plot of benchmark results."""
        lines = []
        lines.append("\nTIME COMPLEXITY PLOT (log scale)")
        lines.append("=" * 60)

        solvers = sorted(set(r.solver_name for r in self.results))
        symbols = ["●", "○", "■", "□", "◆", "◇"]

        # Get all data
        all_data = {}
        max_time = 0
        for i, solver in enumerate(solvers):
            averages = self.get_average_times(solver)
            all_data[solver] = averages
            if averages:
                max_time = max(max_time, max(t for _, t in averages))

        if max_time == 0:
            return "No data to plot"

        # Plot height and width
        height = 15
        width = 50

        # Create plot grid
        import math
        log_max = math.log10(max_time + 1) if max_time > 0 else 1

        for row in range(height, -1, -1):
            if row == height:
                lines.append(f"{max_time:8.1f}ms |")
            elif row == 0:
                lines.append(f"{'0.0':>8}ms |" + "-" * width)
            else:
                time_val = (row / height) * max_time
                line = f"{time_val:8.1f}ms |"

                # Plot points
                chars = [" "] * width
                for i, solver in enumerate(solvers):
                    for n_vars, t in all_data.get(solver, []):
                        # X position based on n_vars
                        all_vars = sorted(set(r.num_vars for r in self.results))
                        if all_vars:
                            x = int((all_vars.index(n_vars) / max(1, len(all_vars) - 1)) * (width - 1))
                            # Y position based on time
                            y = int((t / max_time) * height) if max_time > 0 else 0
                            if y == row and 0 <= x < width:
                                chars[x] = symbols[i % len(symbols)]

                lines.append(line + "".join(chars))

        # X-axis labels
        all_vars = sorted(set(r.num_vars for r in self.results))
        if all_vars:
            x_labels = " " * 10
            for i, n in enumerate(all_vars):
                if i % max(1, len(all_vars) // 5) == 0:
                    pos = int((i / max(1, len(all_vars) - 1)) * width)
                    label = f"n={n}"
                    x_labels = x_labels[:10+pos] + label + x_labels[10+pos+len(label):]
            lines.append(x_labels[:10+width])

        # Legend
        lines.append("\nLegend:")
        for i, solver in enumerate(solvers):
            lines.append(f"  {symbols[i % len(symbols)]} {solver}")

        return "\n".join(lines)


def generate_random_sat(num_vars: int, clause_ratio: float = 3.0,
                        clause_size: int = 3) -> SATInstance:
    """
    Generate a random k-SAT instance.

    Args:
        num_vars: Number of variables
        clause_ratio: Ratio of clauses to variables (e.g., 4.26 for 3-SAT phase transition)
        clause_size: Literals per clause (k in k-SAT)

    Returns:
        Random SAT instance
    """
    num_clauses = int(num_vars * clause_ratio)
    clauses = []

    for _ in range(num_clauses):
        # Pick k random variables without replacement
        vars_in_clause = random.sample(range(1, num_vars + 1),
                                       min(clause_size, num_vars))
        # Randomly negate each variable
        clause = [v if random.random() > 0.5 else -v for v in vars_in_clause]
        clauses.append(clause)

    return SATInstance.from_dimacs(clauses, num_vars)


def generate_satisfiable_sat(num_vars: int, clause_ratio: float = 2.0) -> Tuple[SATInstance, Dict[int, bool]]:
    """
    Generate a random SAT instance that is guaranteed to be satisfiable.

    This works by first generating a random assignment, then creating
    clauses that are all satisfied by this assignment.

    Returns:
        (instance, hidden_assignment)
    """
    # Generate random target assignment
    assignment = {v: random.choice([True, False]) for v in range(1, num_vars + 1)}

    num_clauses = int(num_vars * clause_ratio)
    clauses = []

    for _ in range(num_clauses):
        clause_size = random.randint(2, min(4, num_vars))
        vars_in_clause = random.sample(range(1, num_vars + 1),
                                       min(clause_size, num_vars))

        clause = []
        for v in vars_in_clause:
            # Bias toward matching the assignment (but not always)
            if random.random() < 0.7:
                # Make this literal true under the assignment
                clause.append(v if assignment[v] else -v)
            else:
                clause.append(v if random.random() > 0.5 else -v)

        # Ensure clause is satisfied by flipping one literal if needed
        satisfied = any(
            (lit > 0 and assignment[abs(lit)]) or (lit < 0 and not assignment[abs(lit)])
            for lit in clause
        )
        if not satisfied and clause:
            # Flip one literal to satisfy
            idx = random.randint(0, len(clause) - 1)
            v = abs(clause[idx])
            clause[idx] = v if assignment[v] else -v

        clauses.append(clause)

    return SATInstance.from_dimacs(clauses, num_vars), assignment


def benchmark_solver(solver_fn, instance: SATInstance, solver_name: str) -> BenchmarkResult:
    """Benchmark a single solver on a single instance."""
    # Convert SATInstance to list of int lists for the solver functions
    clauses_list = [[lit.to_int() for lit in c.literals] for c in instance.clauses]

    start = time.perf_counter()
    result_str, _ = solver_fn(clauses_list, instance.num_vars)
    end = time.perf_counter()

    solve_time_ms = (end - start) * 1000

    return BenchmarkResult(
        num_vars=instance.num_vars,
        num_clauses=len(instance.clauses),
        solver_name=solver_name,
        solve_time_ms=solve_time_ms,
        result=result_str
    )


def run_benchmarks(
    var_sizes: List[int] = None,
    samples_per_size: int = 10,
    clause_ratio: float = 2.5,
    solvers: List[str] = None,
    satisfiable_only: bool = False,
    verbose: bool = True
) -> BenchmarkSuite:
    """
    Run comprehensive benchmarks on all solvers.

    Args:
        var_sizes: List of variable counts to test (default: [5,10,15,20,25,30,40,50])
        samples_per_size: Number of random instances per size
        clause_ratio: Clause-to-variable ratio
        solvers: Which solvers to test (default: all)
        satisfiable_only: Only generate satisfiable instances
        verbose: Print progress

    Returns:
        BenchmarkSuite with all results
    """
    if var_sizes is None:
        var_sizes = [5, 10, 15, 20, 25, 30, 40, 50]

    solver_map = {
        "linear": solve_sat_linear,
        "proven": solve_proven,
        "topological": solve_sat,
    }

    if solvers is None:
        solvers = list(solver_map.keys())

    suite = BenchmarkSuite(metadata={
        "timestamp": datetime.now().isoformat(),
        "var_sizes": var_sizes,
        "samples_per_size": samples_per_size,
        "clause_ratio": clause_ratio,
        "solvers": solvers,
        "satisfiable_only": satisfiable_only,
    })

    total_runs = len(var_sizes) * samples_per_size * len(solvers)
    current = 0

    for n_vars in var_sizes:
        for sample_idx in range(samples_per_size):
            # Generate instance
            if satisfiable_only:
                instance, _ = generate_satisfiable_sat(n_vars, clause_ratio)
            else:
                instance = generate_random_sat(n_vars, clause_ratio)

            # Benchmark each solver
            for solver_name in solvers:
                current += 1

                if verbose:
                    pct = (current / total_runs) * 100
                    print(f"\r[{pct:5.1f}%] n={n_vars}, sample {sample_idx+1}, "
                          f"solver={solver_name}...", end="", flush=True)

                solver_fn = solver_map[solver_name]
                result = benchmark_solver(solver_fn, instance, solver_name)
                suite.add(result)

    if verbose:
        print("\nBenchmarks complete!")

    return suite


def quick_benchmark(max_vars: int = 30, samples: int = 5, verbose: bool = True) -> BenchmarkSuite:
    """
    Quick benchmark with smaller sizes for fast testing.

    Args:
        max_vars: Maximum number of variables
        samples: Samples per size
        verbose: Print progress

    Returns:
        BenchmarkSuite with results
    """
    sizes = [n for n in [5, 10, 15, 20, 25, 30, 40, 50] if n <= max_vars]
    return run_benchmarks(
        var_sizes=sizes,
        samples_per_size=samples,
        clause_ratio=2.0,  # Lower ratio for faster solving
        verbose=verbose
    )


def compare_solvers(instance: SATInstance) -> Dict[str, Dict]:
    """
    Compare all solvers on a single instance.

    Returns timing and results for each solver.
    """
    solvers = {
        "linear": solve_sat_linear,
        "proven": solve_proven,
        "topological": solve_sat,
    }

    results = {}
    for name, solver in solvers.items():
        start = time.perf_counter()
        result = solver(instance)
        elapsed = (time.perf_counter() - start) * 1000

        results[name] = {
            "time_ms": elapsed,
            "satisfiable": result.satisfiable,
            "has_assignment": result.assignment is not None,
        }

    return results


if __name__ == "__main__":
    print("topoKEMP2 Benchmark Suite")
    print("=" * 60)

    # Quick benchmark
    print("\nRunning quick benchmark...")
    suite = quick_benchmark(max_vars=30, samples=3)

    # Print summary
    print(suite.print_summary())

    # ASCII plot
    print(suite.plot_ascii())

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(results_dir, f"benchmark_{timestamp}.json")
    suite.save(filepath)
    print(f"\nResults saved to: {filepath}")

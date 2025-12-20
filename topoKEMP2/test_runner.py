"""
Comprehensive Test Runner for topoKEMP2.

This module runs all tests and logs results to the results/ folder.
Each test includes:
- Function call
- Expected result
- Actual result
- Pass/Fail status
- Timing information
"""

import os
import sys
import time
import json
import random
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict

# Ensure we can import topoKEMP2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from topoKEMP2.sat_instance import SATInstance
from topoKEMP2.solver import solve_sat
from topoKEMP2.linear_solver import solve_sat_linear
from topoKEMP2.proven_solver import solve_proven, ProvenSATSolver


@dataclass
class TestResult:
    """Result of a single test."""
    test_name: str
    function_called: str
    input_args: str
    expected_result: str
    actual_result: str
    passed: bool
    time_seconds: float
    error_message: Optional[str] = None
    details: Optional[Dict] = None


class TestLogger:
    """Logger for test results."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        self.results: List[TestResult] = []
        self.start_time = datetime.now()

        os.makedirs(results_dir, exist_ok=True)

    def log(self, result: TestResult):
        """Log a test result."""
        self.results.append(result)
        print(f"{'✓' if result.passed else '✗'} {result.test_name}: {result.actual_result}")

    def save(self, name: str = "sat_solver"):
        """Save all results to file with descriptive name."""
        # Use descriptive name instead of timestamp
        filename = os.path.join(self.results_dir, f"{name}_test_results.json")

        summary = {
            "test_suite": name,
            "timestamp": self.start_time.isoformat(),
            "total_tests": len(self.results),
            "passed": sum(1 for r in self.results if r.passed),
            "failed": sum(1 for r in self.results if not r.passed),
            "pass_rate": f"{100 * sum(1 for r in self.results if r.passed) / len(self.results):.1f}%",
            "results": [asdict(r) for r in self.results]
        }

        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)

        # Also save human-readable summary
        summary_file = os.path.join(self.results_dir, f"{name}_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"topoKEMP2 Test Results\n")
            f.write(f"=" * 60 + "\n")
            f.write(f"Timestamp: {self.start_time.isoformat()}\n")
            f.write(f"Total Tests: {summary['total_tests']}\n")
            f.write(f"Passed: {summary['passed']}\n")
            f.write(f"Failed: {summary['failed']}\n")
            f.write(f"Pass Rate: {100 * summary['passed'] / summary['total_tests']:.1f}%\n")
            f.write(f"=" * 60 + "\n\n")

            for r in self.results:
                status = "PASS" if r.passed else "FAIL"
                f.write(f"[{status}] {r.test_name}\n")
                f.write(f"  Function: {r.function_called}\n")
                f.write(f"  Input: {r.input_args[:100]}...\n" if len(r.input_args) > 100 else f"  Input: {r.input_args}\n")
                f.write(f"  Expected: {r.expected_result}\n")
                f.write(f"  Actual: {r.actual_result}\n")
                f.write(f"  Time: {r.time_seconds:.6f}s\n")
                if r.error_message:
                    f.write(f"  Error: {r.error_message}\n")
                f.write("\n")

        print(f"\nResults saved to: {filename}")
        print(f"Summary saved to: {summary_file}")

        return filename, summary_file


def run_test(logger: TestLogger, test_name: str, func, args: Tuple,
             expected: Any, verifier=None) -> TestResult:
    """Run a single test and log the result."""
    start = time.time()
    error = None
    actual = None
    passed = False

    try:
        actual = func(*args)
        elapsed = time.time() - start

        if verifier:
            passed = verifier(actual, expected, args)
        else:
            passed = actual == expected

    except Exception as e:
        elapsed = time.time() - start
        error = str(e)
        actual = f"ERROR: {e}"
        passed = False

    result = TestResult(
        test_name=test_name,
        function_called=func.__name__ if hasattr(func, '__name__') else str(func),
        input_args=str(args)[:500],
        expected_result=str(expected)[:200],
        actual_result=str(actual)[:200],
        passed=passed,
        time_seconds=elapsed,
        error_message=error
    )

    logger.log(result)
    return result


def verify_sat_result(actual, expected, args):
    """Verify SAT solver result."""
    result, assignment = actual
    expected_result, _ = expected

    if result != expected_result:
        return False

    if result == "SAT" and assignment:
        # Verify the assignment
        clauses = args[0]
        num_vars = args[1] if len(args) > 1 else None
        instance = SATInstance.from_dimacs(clauses, num_vars)
        return instance.evaluate(assignment) is True

    return True


def run_all_tests():
    """Run all tests and save results."""
    logger = TestLogger(os.path.join(os.path.dirname(__file__), "results"))

    print("=" * 60)
    print("topoKEMP2 Comprehensive Test Suite")
    print("=" * 60)

    # ==========================================================================
    # Test Category 1: Basic SAT Tests
    # ==========================================================================
    print("\n[Category 1] Basic SAT Tests")
    print("-" * 40)

    # Test 1.1: Simple satisfiable
    run_test(logger,
             "1.1 Simple SAT",
             solve_sat,
             ([[1, 2], [-1, 2]],),
             ("SAT", None),
             verify_sat_result)

    # Test 1.2: Simple unsatisfiable
    run_test(logger,
             "1.2 Simple UNSAT",
             solve_sat,
             ([[1], [-1]],),
             ("UNSAT", None),
             verify_sat_result)

    # Test 1.3: Empty formula
    run_test(logger,
             "1.3 Empty formula",
             solve_sat,
             ([],),
             ("SAT", None),
             verify_sat_result)

    # Test 1.4: Single variable
    run_test(logger,
             "1.4 Single variable",
             solve_sat,
             ([[1]],),
             ("SAT", None),
             verify_sat_result)

    # ==========================================================================
    # Test Category 2: Linear Solver Tests
    # ==========================================================================
    print("\n[Category 2] Linear Solver Tests")
    print("-" * 40)

    # Test 2.1: 2-SAT instance
    run_test(logger,
             "2.1 2-SAT instance",
             solve_sat_linear,
             ([[1, 2], [-1, 3], [-2, -3]],),
             ("SAT", None),
             verify_sat_result)

    # Test 2.2: Unit propagation
    run_test(logger,
             "2.2 Unit propagation",
             solve_sat_linear,
             ([[1], [1, 2], [-1, 2]],),
             ("SAT", None),
             verify_sat_result)

    # Test 2.3: Conflict detection
    run_test(logger,
             "2.3 Conflict detection",
             solve_sat_linear,
             ([[1], [-1]],),
             ("UNSAT", None),
             verify_sat_result)

    # ==========================================================================
    # Test Category 3: Proven Solver Tests
    # ==========================================================================
    print("\n[Category 3] Proven Solver Tests")
    print("-" * 40)

    # Test 3.1: Proven solver basic
    run_test(logger,
             "3.1 Proven solver basic",
             solve_proven,
             ([[1, 2], [-1, 2]],),
             ("SAT", None),
             verify_sat_result)

    # Test 3.2: Proven solver UNSAT
    run_test(logger,
             "3.2 Proven solver UNSAT",
             solve_proven,
             ([[1], [-1]],),
             ("UNSAT", None),
             verify_sat_result)

    # Test 3.3: Proven solver complex
    clauses_complex = [
        [1, 2, 3],
        [-1, -2],
        [2, -3],
        [-1, 3],
        [1, -2, -3]
    ]
    run_test(logger,
             "3.3 Proven solver complex",
             solve_proven,
             (clauses_complex,),
             ("SAT", None),
             verify_sat_result)

    # ==========================================================================
    # Test Category 4: Random SAT Tests
    # ==========================================================================
    print("\n[Category 4] Random SAT Tests")
    print("-" * 40)

    random.seed(42)

    for i in range(5):
        n_vars = 10
        n_clauses = int(3.5 * n_vars)  # Below phase transition

        clauses = []
        for _ in range(n_clauses):
            clause = random.sample(range(1, n_vars + 1), 3)
            clause = [v if random.random() > 0.5 else -v for v in clause]
            clauses.append(clause)

        run_test(logger,
                 f"4.{i+1} Random 3-SAT (n={n_vars}, m={n_clauses})",
                 solve_sat,
                 (clauses, n_vars),
                 ("SAT", None),  # Most should be SAT at this ratio
                 verify_sat_result)

    # ==========================================================================
    # Test Category 5: Edge Cases
    # ==========================================================================
    print("\n[Category 5] Edge Cases")
    print("-" * 40)

    # Test 5.1: Tautology clause
    run_test(logger,
             "5.1 Tautology in clause",
             solve_sat,
             ([[1, -1]], 1),
             ("SAT", None),
             verify_sat_result)

    # Test 5.2: All positive literals
    run_test(logger,
             "5.2 All positive literals",
             solve_sat,
             ([[1, 2, 3], [1, 2], [2, 3]], 3),
             ("SAT", None),
             verify_sat_result)

    # Test 5.3: All negative literals
    run_test(logger,
             "5.3 All negative literals",
             solve_sat,
             ([[-1, -2, -3], [-1, -2], [-2, -3]], 3),
             ("SAT", None),
             verify_sat_result)

    # Test 5.4: Large clause
    run_test(logger,
             "5.4 Large clause",
             solve_sat,
             ([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], 10),
             ("SAT", None),
             verify_sat_result)

    # ==========================================================================
    # Test Category 6: Performance Tests
    # ==========================================================================
    print("\n[Category 6] Performance Tests")
    print("-" * 40)

    for n_vars in [20, 30, 40]:
        # Use lower clause ratio for larger instances to ensure SAT
        ratio = 3.0 if n_vars >= 40 else 3.5

        clauses = []
        for _ in range(int(ratio * n_vars)):
            clause = random.sample(range(1, n_vars + 1), 3)
            clause = [v if random.random() > 0.5 else -v for v in clause]
            clauses.append(clause)

        run_test(logger,
                 f"6.{n_vars//10} Performance n={n_vars}",
                 solve_sat_linear,
                 (clauses, n_vars),
                 ("SAT", None),
                 verify_sat_result)

    # ==========================================================================
    # Test Category 7: Solver Comparison
    # ==========================================================================
    print("\n[Category 7] Solver Comparison")
    print("-" * 40)

    test_clauses = [
        [1, 2, 3],
        [-1, 2],
        [1, -2, 3],
        [-3, 2]
    ]

    run_test(logger,
             "7.1 solve_sat on test instance",
             solve_sat,
             (test_clauses, 3),
             ("SAT", None),
             verify_sat_result)

    run_test(logger,
             "7.2 solve_sat_linear on test instance",
             solve_sat_linear,
             (test_clauses, 3),
             ("SAT", None),
             verify_sat_result)

    run_test(logger,
             "7.3 solve_proven on test instance",
             solve_proven,
             (test_clauses, 3),
             ("SAT", None),
             verify_sat_result)

    # ==========================================================================
    # Save Results
    # ==========================================================================
    print("\n" + "=" * 60)
    json_file, summary_file = logger.save("sat_solver")

    total = len(logger.results)
    passed = sum(1 for r in logger.results if r.passed)
    failed = total - passed

    print(f"\nFinal Results: {passed}/{total} passed ({100*passed/total:.1f}%)")

    if failed > 0:
        print(f"\nFailed tests:")
        for r in logger.results:
            if not r.passed:
                print(f"  - {r.test_name}: {r.actual_result}")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

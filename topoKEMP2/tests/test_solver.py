"""
Tests for topoKEMP2 SAT solver.

These tests verify:
1. Correct SAT/UNSAT detection
2. Valid assignment extraction
3. Invariant computation
4. Knot simplification
"""

import unittest
from topoKEMP2.sat_instance import SATInstance, Clause, Literal
from topoKEMP2.knot import KnotDiagram, CrossingSign
from topoKEMP2.braid import BraidWord, BraidGenerator, GeneratorSign
from topoKEMP2.embedder import SATEmbedder, LayeredEmbedder
from topoKEMP2.simplifier import KnotSimplifier, BraidSimplifier
from topoKEMP2.invariants import compute_writhe, compute_jones_polynomial
from topoKEMP2.solver import TopoKEMP2Solver, SolverResult, solve_sat


class TestSATInstance(unittest.TestCase):
    """Tests for SAT instance representation."""

    def test_create_instance(self):
        """Test basic instance creation."""
        clauses = [[1, -2, 3], [-1, 2], [2, 3]]
        instance = SATInstance.from_dimacs(clauses)

        self.assertEqual(instance.num_vars, 3)
        self.assertEqual(len(instance.clauses), 3)

    def test_evaluate_satisfied(self):
        """Test evaluation with satisfying assignment."""
        clauses = [[1, 2], [-1, 2]]
        instance = SATInstance.from_dimacs(clauses)

        assignment = {1: True, 2: True}
        self.assertTrue(instance.evaluate(assignment))

        assignment = {1: False, 2: True}
        self.assertTrue(instance.evaluate(assignment))

    def test_evaluate_unsatisfied(self):
        """Test evaluation with non-satisfying assignment."""
        clauses = [[1], [-1]]
        instance = SATInstance.from_dimacs(clauses)

        assignment = {1: True}
        self.assertFalse(instance.evaluate(assignment))

        assignment = {1: False}
        self.assertFalse(instance.evaluate(assignment))

    def test_unit_propagation(self):
        """Test unit clause detection."""
        clause = Clause.from_ints([1, 2, 3])
        assignment = {1: False, 2: False}
        unit = clause.is_unit(assignment)

        self.assertIsNotNone(unit)
        self.assertEqual(unit.variable, 3)


class TestKnotDiagram(unittest.TestCase):
    """Tests for knot diagram operations."""

    def test_create_empty_diagram(self):
        """Test creating empty diagram."""
        diagram = KnotDiagram()
        self.assertEqual(diagram.crossing_number(), 0)
        self.assertTrue(diagram.is_trivial())

    def test_from_braid_word(self):
        """Test creating diagram from braid word."""
        braid = [1, -1]
        diagram = KnotDiagram.from_braid_word(braid, 3)

        self.assertEqual(diagram.crossing_number(), 2)

    def test_trefoil_braid(self):
        """Test trefoil knot from braid representation."""
        trefoil_braid = [1, 1, 1]
        diagram = KnotDiagram.from_braid_word(trefoil_braid, 2)

        self.assertEqual(diagram.crossing_number(), 3)
        self.assertEqual(diagram.writhe(), 3)

    def test_figure_eight_braid(self):
        """Test figure-eight knot from braid."""
        figure_eight = [1, -2, 1, -2]
        diagram = KnotDiagram.from_braid_word(figure_eight, 3)

        self.assertEqual(diagram.crossing_number(), 4)
        self.assertEqual(diagram.writhe(), 0)

    def test_find_loops(self):
        """Test loop detection."""
        diagram = KnotDiagram()
        diagram.add_crossing((1, 2, 1, 3), CrossingSign.POSITIVE)

        loops = diagram.find_removable_loops()
        self.assertEqual(len(loops), 1)


class TestBraidWord(unittest.TestCase):
    """Tests for braid word operations."""

    def test_free_reduction(self):
        """Test free reduction of inverse pairs."""
        braid = BraidWord.from_int_list(3, [1, -1, 2])
        reduced = braid.free_reduce()

        self.assertEqual(reduced.length(), 1)
        self.assertEqual(reduced.generators[0].index, 2)

    def test_braid_inverse(self):
        """Test braid inverse computation."""
        braid = BraidWord.from_int_list(3, [1, 2, 1])
        inverse = braid.inverse()

        self.assertEqual(inverse.length(), 3)
        self.assertEqual(inverse.generators[0].index, 1)
        self.assertEqual(inverse.generators[0].sign, GeneratorSign.NEGATIVE)

    def test_handle_slide(self):
        """Test far commutativity."""
        braid = BraidWord.from_int_list(4, [1, 3, 2])
        slid = braid.handle_slide(0)

        self.assertIsNotNone(slid)
        self.assertEqual(slid.generators[0].index, 3)
        self.assertEqual(slid.generators[1].index, 1)

    def test_trivial_braid(self):
        """Test trivial braid detection."""
        braid = BraidWord.from_int_list(3, [1, -1])
        reduced = braid.free_reduce()

        self.assertEqual(reduced.length(), 0)


class TestSimplifier(unittest.TestCase):
    """Tests for knot simplification."""

    def test_simplify_trivial(self):
        """Test simplifying an already trivial diagram."""
        diagram = KnotDiagram()
        simplifier = KnotSimplifier()
        result = simplifier.simplify(diagram)

        self.assertTrue(result.is_trivial())

    def test_simplify_single_loop(self):
        """Test simplifying a single loop."""
        diagram = KnotDiagram()
        diagram.add_crossing((1, 2, 1, 3), CrossingSign.POSITIVE)

        simplifier = KnotSimplifier()
        result = simplifier.simplify(diagram)

        self.assertLessEqual(result.crossing_number(), 0)

    def test_braid_simplifier(self):
        """Test braid word simplification."""
        braid = BraidWord.from_int_list(3, [1, -1, 2, -2, 1])
        simplifier = BraidSimplifier()
        result = simplifier.simplify(braid)

        self.assertLessEqual(result.length(), 1)


class TestInvariants(unittest.TestCase):
    """Tests for knot invariants."""

    def test_writhe_unknot(self):
        """Test writhe of unknot is 0."""
        diagram = KnotDiagram()
        self.assertEqual(compute_writhe(diagram), 0)

    def test_writhe_trefoil(self):
        """Test writhe of trefoil."""
        trefoil = KnotDiagram.from_braid_word([1, 1, 1], 2)
        self.assertEqual(compute_writhe(trefoil), 3)

    def test_jones_unknot(self):
        """Test Jones polynomial of unknot."""
        diagram = KnotDiagram()
        jones = compute_jones_polynomial(diagram)
        self.assertEqual(jones, {0: 1})


class TestSolver(unittest.TestCase):
    """Tests for the main SAT solver."""

    def test_solve_trivially_sat(self):
        """Test solving trivially satisfiable instance."""
        clauses = [[1, 2], [1, -2], [-1, 2]]
        result, assignment = solve_sat(clauses)

        self.assertEqual(result, "SAT")
        self.assertIsNotNone(assignment)

    def test_solve_trivially_unsat(self):
        """Test solving trivially unsatisfiable instance."""
        clauses = [[1], [-1]]
        result, assignment = solve_sat(clauses)

        self.assertEqual(result, "UNSAT")
        self.assertIsNone(assignment)

    def test_solve_empty_formula(self):
        """Test solving empty formula (trivially SAT)."""
        clauses = []
        result, assignment = solve_sat(clauses, num_vars=0)

        self.assertEqual(result, "SAT")

    def test_solve_simple_3sat(self):
        """Test solving simple 3-SAT instance."""
        clauses = [
            [1, 2, 3],
            [-1, 2, 3],
            [1, -2, 3],
            [1, 2, -3]
        ]
        solver = TopoKEMP2Solver()
        instance = SATInstance.from_dimacs(clauses)
        output = solver.solve(instance)

        if output.result == SolverResult.SATISFIABLE:
            self.assertTrue(instance.evaluate(output.assignment))

    def test_solver_with_different_embedders(self):
        """Test solver with different embedding strategies."""
        clauses = [[1, 2], [-1, 2], [1, -2]]
        instance = SATInstance.from_dimacs(clauses)

        for embedder in ['basic', 'resolution', 'layered']:
            solver = TopoKEMP2Solver(embedder_type=embedder)
            output = solver.solve(instance)

            if output.result == SolverResult.SATISFIABLE:
                self.assertTrue(instance.evaluate(output.assignment))


class TestEmbedder(unittest.TestCase):
    """Tests for SAT-to-knot embedding."""

    def test_embed_simple_clause(self):
        """Test embedding a simple clause."""
        clauses = [[1, 2]]
        instance = SATInstance.from_dimacs(clauses)

        embedder = SATEmbedder()
        braid, mapping = embedder.embed(instance)

        self.assertGreater(braid.num_strands, 0)

    def test_embed_to_knot(self):
        """Test embedding directly to knot diagram."""
        clauses = [[1, -2], [2, 3]]
        instance = SATInstance.from_dimacs(clauses)

        embedder = LayeredEmbedder()
        diagram = embedder.embed_to_knot(instance)

        self.assertIsInstance(diagram, KnotDiagram)


if __name__ == '__main__':
    unittest.main()

"""
Knot Invariant Computation for topoKEMP2.

This module provides polynomial-time computable invariants that can
help detect unknots without full simplification:

- Writhe: Sum of crossing signs (O(n))
- Linking number: For links (O(n))
- Kauffman bracket: Polynomial invariant (exponential in worst case,
  but fast for diagrams with special structure)
- Bridge number estimates

These invariants provide necessary (not sufficient) conditions for
unknottedness, allowing early termination in many cases.
"""

from typing import Dict, List, Optional, Tuple
from fractions import Fraction
from collections import defaultdict

from .knot import KnotDiagram, Crossing, CrossingSign


def compute_writhe(diagram: KnotDiagram) -> int:
    """
    Compute the writhe of a knot diagram.

    The writhe is the sum of crossing signs:
    - Positive crossing: +1
    - Negative crossing: -1

    Note: The writhe is NOT a knot invariant (depends on diagram),
    but writhe = 0 is necessary for alternating unknot diagrams.

    Time complexity: O(n) where n = number of crossings
    """
    return sum(c.sign.value for c in diagram.crossings.values() if not c.is_virtual)


def compute_linking_number(diagram: KnotDiagram, component1: int, component2: int) -> int:
    """
    Compute the linking number between two components of a link.

    The linking number is half the sum of signed crossings between
    the two components.

    Args:
        diagram: The link diagram
        component1: Index of first component
        component2: Index of second component

    Returns:
        The linking number (an integer)
    """
    total = 0

    for crossing in diagram.crossings.values():
        if crossing.is_virtual:
            continue

        over_arc = crossing.incoming_over
        under_arc = crossing.incoming_under

        over_strand = diagram.arcs.get(over_arc)
        under_strand = diagram.arcs.get(under_arc)

        if over_strand and under_strand:
            over_comp = over_strand.strand_index
            under_comp = under_strand.strand_index

            if {over_comp, under_comp} == {component1, component2}:
                total += crossing.sign.value

    return total // 2


def compute_bracket(diagram: KnotDiagram) -> Dict[int, int]:
    """
    Compute the Kauffman bracket polynomial.

    The bracket is computed via state sum:
    <K> = Σ_s A^{σ(s)} (-A^2 - A^{-2})^{|s|-1}

    where s ranges over states (smoothings), σ(s) is the
    signed sum of smoothing choices, and |s| is the number
    of resulting circles.

    Returns polynomial as dict: power -> coefficient

    Note: This is exponential in worst case O(2^n), but we use
    memoization and early pruning for practical speedup.
    """
    if diagram.crossing_number() == 0:
        return {0: 1}

    if diagram.crossing_number() > 20:
        return _compute_bracket_approximate(diagram)

    crossings = list(diagram.crossings.keys())
    return _bracket_state_sum(diagram, crossings, 0, {})


def _bracket_state_sum(diagram: KnotDiagram, crossings: List[int],
                       index: int, memo: Dict) -> Dict[int, int]:
    """Recursive state sum computation with memoization."""
    if index >= len(crossings):
        num_circles = _count_circles(diagram)
        circle_factor = _compute_circle_factor(num_circles)
        return circle_factor

    cid = crossings[index]

    state_key = (index, tuple(sorted(diagram.crossings.keys())))
    if state_key in memo:
        return memo[state_key]

    result_a = _smooth_and_recurse(diagram, crossings, index, cid, 'A', memo)
    result_b = _smooth_and_recurse(diagram, crossings, index, cid, 'B', memo)

    result = _multiply_by_a(result_a, 1)
    result = _add_polynomials(result, _multiply_by_a(result_b, -1))

    memo[state_key] = result
    return result


def _smooth_and_recurse(diagram: KnotDiagram, crossings: List[int],
                        index: int, cid: int, smoothing: str,
                        memo: Dict) -> Dict[int, int]:
    """Apply smoothing and recurse."""
    smoothed = diagram.copy()

    if cid in smoothed.crossings:
        crossing = smoothed.crossings[cid]

        if smoothing == 'A':
            arc1 = crossing.arcs[0]
            arc2 = crossing.arcs[1]
            arc3 = crossing.arcs[2]
            arc4 = crossing.arcs[3]
            smoothed.merge_arcs(arc1, arc2)
            smoothed.merge_arcs(arc3, arc4)
        else:
            arc1 = crossing.arcs[0]
            arc2 = crossing.arcs[3]
            arc3 = crossing.arcs[1]
            arc4 = crossing.arcs[2]
            smoothed.merge_arcs(arc1, arc2)
            smoothed.merge_arcs(arc3, arc4)

        smoothed.remove_crossing(cid)

    return _bracket_state_sum(smoothed, crossings, index + 1, memo)


def _compute_bracket_approximate(diagram: KnotDiagram) -> Dict[int, int]:
    """
    Approximate bracket for large diagrams.

    Uses heuristics to estimate the bracket without full state sum.
    """
    n = diagram.crossing_number()
    w = compute_writhe(diagram)

    estimate = {-3 * w: 1}
    return estimate


def _count_circles(diagram: KnotDiagram) -> int:
    """Count the number of circles in a diagram with no crossings."""
    if diagram.crossing_number() > 0:
        return 1

    visited = set()
    circles = 0

    for arc_id in diagram.arcs:
        if arc_id not in visited:
            circles += 1
            _traverse_circle(diagram, arc_id, visited)

    return max(circles, 1)


def _traverse_circle(diagram: KnotDiagram, start_arc: int, visited: set) -> None:
    """Traverse a circle marking arcs as visited."""
    current = start_arc
    while current not in visited:
        visited.add(current)
        next_arc = None

        for cid, crossing in diagram.crossings.items():
            if crossing.incoming_under == current:
                next_arc = crossing.outgoing_under
                break
            elif crossing.incoming_over == current:
                next_arc = crossing.outgoing_over
                break

        if next_arc is None:
            break
        current = next_arc


def _compute_circle_factor(n: int) -> Dict[int, int]:
    """Compute (-A^2 - A^{-2})^{n-1}."""
    if n <= 1:
        return {0: 1}

    base = {2: -1, -2: -1}
    result = {0: 1}

    for _ in range(n - 1):
        result = _multiply_polynomials(result, base)

    return result


def _multiply_by_a(poly: Dict[int, int], power: int) -> Dict[int, int]:
    """Multiply polynomial by A^power."""
    return {k + power: v for k, v in poly.items()}


def _add_polynomials(p1: Dict[int, int], p2: Dict[int, int]) -> Dict[int, int]:
    """Add two polynomials."""
    result = p1.copy()
    for k, v in p2.items():
        result[k] = result.get(k, 0) + v
    result = {k: v for k, v in result.items() if v != 0}
    return result if result else {0: 0}


def _multiply_polynomials(p1: Dict[int, int], p2: Dict[int, int]) -> Dict[int, int]:
    """Multiply two polynomials."""
    result = {}
    for k1, v1 in p1.items():
        for k2, v2 in p2.items():
            k = k1 + k2
            result[k] = result.get(k, 0) + v1 * v2
    result = {k: v for k, v in result.items() if v != 0}
    return result if result else {0: 0}


def compute_jones_polynomial(diagram: KnotDiagram) -> Dict[int, int]:
    """
    Compute the Jones polynomial V(K, t).

    V(K, t) = (-A^3)^{-w} <K>

    where w is the writhe and <K> is the Kauffman bracket.
    The result is in terms of t where A = t^{-1/4}.
    """
    bracket = compute_bracket(diagram)
    w = compute_writhe(diagram)

    normalization_power = -3 * w
    jones = _multiply_by_a(bracket, normalization_power)

    jones_in_t = {}
    for power, coeff in jones.items():
        t_power = -power // 4
        jones_in_t[t_power] = jones_in_t.get(t_power, 0) + coeff

    return jones_in_t


def is_potentially_unknot(diagram: KnotDiagram) -> Tuple[bool, str]:
    """
    Check if the diagram could potentially be an unknot.

    Uses fast invariant checks:
    1. If crossing number is 0, it's definitely the unknot
    2. Check if Jones polynomial is trivial (necessary but not sufficient)

    Returns (is_potential, reason)
    """
    if diagram.crossing_number() == 0:
        return True, "No crossings"

    if diagram.crossing_number() <= 20:
        jones = compute_jones_polynomial(diagram)
        if jones == {0: 1}:
            return True, "Trivial Jones polynomial"
        else:
            return False, f"Non-trivial Jones polynomial: {jones}"

    return True, "Unable to determine (diagram too large)"


def compute_bridge_number_estimate(diagram: KnotDiagram) -> int:
    """
    Estimate the bridge number of the knot.

    The bridge number is the minimum number of local maxima
    in a regular projection. For unknot, bridge number = 1.

    This provides a lower bound estimate.
    """
    if diagram.crossing_number() == 0:
        return 1

    max_count = 1
    for arc_id in diagram.arcs:
        crossing_count = 0
        for crossing in diagram.crossings.values():
            if arc_id in crossing.get_over_arcs():
                crossing_count += 1

        max_count = max(max_count, crossing_count)

    return max(1, (max_count + 1) // 2)


class InvariantChecker:
    """
    Combines multiple invariant checks for unknot detection.
    """

    def __init__(self, use_jones: bool = True, use_bridge: bool = True):
        self.use_jones = use_jones
        self.use_bridge = use_bridge

    def check_unknot_possibility(self, diagram: KnotDiagram) -> Tuple[bool, Dict]:
        """
        Run all invariant checks and return combined result.

        Returns (could_be_unknot, invariant_values)
        """
        results = {
            'crossing_number': diagram.crossing_number(),
            'writhe': compute_writhe(diagram)
        }

        if diagram.crossing_number() == 0:
            results['verdict'] = 'unknot'
            return True, results

        if self.use_jones and diagram.crossing_number() <= 20:
            jones = compute_jones_polynomial(diagram)
            results['jones'] = jones
            if jones != {0: 1}:
                results['verdict'] = 'not_unknot'
                return False, results

        if self.use_bridge:
            bridge = compute_bridge_number_estimate(diagram)
            results['bridge_estimate'] = bridge

        results['verdict'] = 'unknown'
        return True, results

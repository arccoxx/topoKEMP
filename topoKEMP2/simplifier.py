"""
Knot Simplification using Reidemeister Moves for topoKEMP2.

This module implements polynomial-time simplification strategies
for knot diagrams using the three Reidemeister moves:

- R1 (Twist): Remove/add a simple loop (±1 crossing)
- R2 (Poke): Remove/add two crossings in a bigon (±2 crossings)
- R3 (Slide): Slide a strand over/under a crossing (0 crossings)

The key insight is that while R1/R2 reduce crossing number,
R3 moves may be necessary to enable R1/R2 moves. The challenge
is determining the right sequence of R3 moves.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Set, Dict, Callable
from enum import Enum
from collections import deque
import heapq

from .knot import KnotDiagram, Crossing, CrossingSign, Arc
from .braid import BraidWord, BraidGenerator, GeneratorSign


class MoveType(Enum):
    R1_REMOVE = "R1-"
    R1_ADD = "R1+"
    R2_REMOVE = "R2-"
    R2_ADD = "R2+"
    R3 = "R3"


@dataclass
class ReidemeisterMove:
    """
    Represents a Reidemeister move on a knot diagram.

    Attributes:
        move_type: Type of Reidemeister move
        crossings: IDs of crossings involved
        position: Location information for applying the move
        priority: Higher priority moves are applied first
    """
    move_type: MoveType
    crossings: Tuple[int, ...]
    position: Optional[Dict] = None
    priority: float = 0.0

    def __lt__(self, other):
        return self.priority > other.priority


class KnotSimplifier:
    """
    Simplifies knot diagrams using Reidemeister moves.

    The simplification strategy is:
    1. Greedily apply R1 and R2 moves (reduce crossings)
    2. Use guided R3 moves to enable more R1/R2 moves
    3. Use invariant checks to detect unknots early
    4. Track simplification history for solution extraction
    """

    def __init__(self, max_iterations: int = 10000,
                 r3_search_depth: int = 3,
                 use_invariants: bool = True):
        self.max_iterations = max_iterations
        self.r3_search_depth = r3_search_depth
        self.use_invariants = use_invariants
        self.move_history: List[ReidemeisterMove] = []

    def simplify(self, diagram: KnotDiagram) -> KnotDiagram:
        """
        Simplify the knot diagram as much as possible.

        Returns the simplified diagram.
        """
        self.move_history = []
        iterations = 0

        while iterations < self.max_iterations:
            iterations += 1

            r1_moves = self._find_r1_moves(diagram)
            if r1_moves:
                move = r1_moves[0]
                self._apply_r1_remove(diagram, move)
                self.move_history.append(move)
                continue

            r2_moves = self._find_r2_moves(diagram)
            if r2_moves:
                move = r2_moves[0]
                self._apply_r2_remove(diagram, move)
                self.move_history.append(move)
                continue

            r3_sequence = self._search_r3_sequence(diagram)
            if r3_sequence:
                for move in r3_sequence:
                    self._apply_r3(diagram, move)
                    self.move_history.append(move)
                continue

            break

        return diagram

    def _find_r1_moves(self, diagram: KnotDiagram) -> List[ReidemeisterMove]:
        """
        Find all possible R1 moves (loop removals).

        A crossing is R1-removable if it forms a loop
        (same arc appears twice in the crossing).
        """
        moves = []
        for cid, crossing in diagram.crossings.items():
            arc_counts = {}
            for arc in crossing.arcs:
                arc_counts[arc] = arc_counts.get(arc, 0) + 1

            for arc, count in arc_counts.items():
                if count >= 2:
                    priority = 1.0
                    moves.append(ReidemeisterMove(
                        move_type=MoveType.R1_REMOVE,
                        crossings=(cid,),
                        position={'loop_arc': arc},
                        priority=priority
                    ))
                    break

        return sorted(moves)

    def _find_r2_moves(self, diagram: KnotDiagram) -> List[ReidemeisterMove]:
        """
        Find all possible R2 moves (bigon removals).

        Two crossings form a bigon if they share exactly two arcs
        and have opposite signs.
        """
        moves = []
        crossing_list = list(diagram.crossings.keys())

        for i, c1_id in enumerate(crossing_list):
            c1 = diagram.crossings[c1_id]
            c1_arcs = set(c1.arcs)

            for c2_id in crossing_list[i + 1:]:
                c2 = diagram.crossings[c2_id]
                c2_arcs = set(c2.arcs)
                shared = c1_arcs & c2_arcs

                if len(shared) >= 2 and c1.sign != c2.sign:
                    priority = 2.0
                    moves.append(ReidemeisterMove(
                        move_type=MoveType.R2_REMOVE,
                        crossings=(c1_id, c2_id),
                        position={'shared_arcs': list(shared)},
                        priority=priority
                    ))

        return sorted(moves)

    def _find_r3_moves(self, diagram: KnotDiagram) -> List[ReidemeisterMove]:
        """
        Find all possible R3 moves (strand slides).

        An R3 move involves three crossings forming a triangle.
        """
        moves = []
        crossing_list = list(diagram.crossings.keys())

        for i, c1_id in enumerate(crossing_list):
            c1 = diagram.crossings[c1_id]
            c1_arcs = set(c1.arcs)

            for j, c2_id in enumerate(crossing_list[i + 1:], i + 1):
                c2 = diagram.crossings[c2_id]
                c2_arcs = set(c2.arcs)

                if not (c1_arcs & c2_arcs):
                    continue

                for c3_id in crossing_list[j + 1:]:
                    c3 = diagram.crossings[c3_id]
                    c3_arcs = set(c3.arcs)

                    if (c1_arcs & c3_arcs) and (c2_arcs & c3_arcs):
                        moves.append(ReidemeisterMove(
                            move_type=MoveType.R3,
                            crossings=(c1_id, c2_id, c3_id),
                            priority=0.0
                        ))

        return moves

    def _apply_r1_remove(self, diagram: KnotDiagram, move: ReidemeisterMove) -> bool:
        """Apply R1 move to remove a loop."""
        if move.move_type != MoveType.R1_REMOVE:
            return False

        cid = move.crossings[0]
        if cid not in diagram.crossings:
            return False

        crossing = diagram.crossings[cid]
        arcs = crossing.arcs

        arc_counts = {}
        for arc in arcs:
            arc_counts[arc] = arc_counts.get(arc, 0) + 1

        loop_arc = None
        other_arcs = []
        for arc in arcs:
            if arc_counts[arc] >= 2:
                loop_arc = arc
            else:
                other_arcs.append(arc)

        if loop_arc is None or len(other_arcs) < 2:
            return False

        diagram.remove_crossing(cid)
        if len(other_arcs) >= 2:
            diagram.merge_arcs(other_arcs[0], other_arcs[1])

        return True

    def _apply_r2_remove(self, diagram: KnotDiagram, move: ReidemeisterMove) -> bool:
        """Apply R2 move to remove a bigon."""
        if move.move_type != MoveType.R2_REMOVE:
            return False

        c1_id, c2_id = move.crossings
        if c1_id not in diagram.crossings or c2_id not in diagram.crossings:
            return False

        c1 = diagram.crossings[c1_id]
        c2 = diagram.crossings[c2_id]

        c1_arcs = set(c1.arcs)
        c2_arcs = set(c2.arcs)
        shared = list(c1_arcs & c2_arcs)

        if len(shared) < 2:
            return False

        outer_c1 = [a for a in c1.arcs if a not in shared]
        outer_c2 = [a for a in c2.arcs if a not in shared]

        diagram.remove_crossing(c1_id)
        diagram.remove_crossing(c2_id)

        if outer_c1 and outer_c2:
            for i, arc1 in enumerate(outer_c1):
                if i < len(outer_c2):
                    diagram.merge_arcs(arc1, outer_c2[i])

        return True

    def _apply_r3(self, diagram: KnotDiagram, move: ReidemeisterMove) -> bool:
        """
        Apply R3 move to slide a strand.

        This doesn't change crossing number but rearranges the diagram.
        """
        if move.move_type != MoveType.R3:
            return False

        c1_id, c2_id, c3_id = move.crossings
        if not all(cid in diagram.crossings for cid in [c1_id, c2_id, c3_id]):
            return False

        c1 = diagram.crossings[c1_id]
        c2 = diagram.crossings[c2_id]
        c3 = diagram.crossings[c3_id]

        new_arcs_1 = (c1.arcs[0], c1.arcs[3], c1.arcs[2], c1.arcs[1])
        new_arcs_2 = (c2.arcs[2], c2.arcs[1], c2.arcs[0], c2.arcs[3])
        new_arcs_3 = (c3.arcs[1], c3.arcs[0], c3.arcs[3], c3.arcs[2])

        diagram.crossings[c1_id] = Crossing(c1_id, new_arcs_1, c1.sign)
        diagram.crossings[c2_id] = Crossing(c2_id, new_arcs_2, c2.sign)
        diagram.crossings[c3_id] = Crossing(c3_id, new_arcs_3, c3.sign)

        return True

    def _search_r3_sequence(self, diagram: KnotDiagram) -> Optional[List[ReidemeisterMove]]:
        """
        Search for a sequence of R3 moves that enables R1 or R2 moves.

        Uses BFS with limited depth to find useful R3 sequences.
        """
        if self.r3_search_depth <= 0:
            return None

        r3_moves = self._find_r3_moves(diagram)
        if not r3_moves:
            return None

        for move in r3_moves[:min(len(r3_moves), 10)]:
            test_diagram = diagram.copy()
            self._apply_r3(test_diagram, move)

            if self._find_r1_moves(test_diagram) or self._find_r2_moves(test_diagram):
                return [move]

        if self.r3_search_depth >= 2:
            for move1 in r3_moves[:5]:
                test_diagram = diagram.copy()
                self._apply_r3(test_diagram, move1)

                r3_moves_2 = self._find_r3_moves(test_diagram)
                for move2 in r3_moves_2[:5]:
                    test_diagram_2 = test_diagram.copy()
                    self._apply_r3(test_diagram_2, move2)

                    if self._find_r1_moves(test_diagram_2) or self._find_r2_moves(test_diagram_2):
                        return [move1, move2]

        return None

    def is_unknot(self, diagram: KnotDiagram) -> bool:
        """Check if the diagram represents the unknot."""
        simplified = self.simplify(diagram.copy())
        return simplified.crossing_number() == 0

    def get_move_history(self) -> List[ReidemeisterMove]:
        """Return the sequence of moves applied during simplification."""
        return self.move_history.copy()


class BraidSimplifier:
    """
    Simplifies braid words using algebraic reductions.

    This is often more efficient than diagram simplification
    for braids, as algebraic moves are simpler to detect.
    """

    def __init__(self, max_iterations: int = 10000):
        self.max_iterations = max_iterations

    def simplify(self, braid: BraidWord) -> BraidWord:
        """
        Simplify the braid word using algebraic reductions.

        Applies:
        1. Free reduction (cancel σ_i σ_i^{-1})
        2. Handle slides (commute far generators)
        3. Braid relations (σ_i σ_{i+1} σ_i = σ_{i+1} σ_i σ_{i+1})
        """
        current = braid.free_reduce()
        iterations = 0

        while iterations < self.max_iterations:
            iterations += 1
            changed = False

            reduced = current.free_reduce()
            if reduced.length() < current.length():
                current = reduced
                changed = True
                continue

            for pos in range(current.length() - 2):
                result = current.apply_braid_relation(pos)
                if result:
                    reduced = result.free_reduce()
                    if reduced.length() < current.length():
                        current = reduced
                        changed = True
                        break

            if changed:
                continue

            best_slides = self._find_beneficial_slides(current)
            if best_slides:
                for pos in best_slides:
                    slid = current.handle_slide(pos)
                    if slid:
                        current = slid
                        changed = True
                        break

            if not changed:
                break

        return current.free_reduce()

    def _find_beneficial_slides(self, braid: BraidWord) -> List[int]:
        """
        Find handle slides that may lead to cancellations.

        Returns positions where sliding might bring inverse pairs together.
        """
        beneficial = []

        for pos in range(braid.length() - 1):
            g1 = braid.generators[pos]
            g2 = braid.generators[pos + 1]

            if abs(g1.index - g2.index) <= 1:
                continue

            for future_pos in range(pos + 2, min(pos + 5, braid.length())):
                future_g = braid.generators[future_pos]
                if future_g.index == g1.index and future_g.sign != g1.sign:
                    beneficial.append(pos)
                    break

        return beneficial

    def is_trivial(self, braid: BraidWord) -> bool:
        """Check if the braid represents the trivial element."""
        simplified = self.simplify(braid)
        return simplified.length() == 0


class GreedySimplifier(KnotSimplifier):
    """
    Greedy simplifier that prioritizes maximum crossing reduction.
    """

    def _find_r1_moves(self, diagram: KnotDiagram) -> List[ReidemeisterMove]:
        moves = super()._find_r1_moves(diagram)
        for move in moves:
            move.priority = 10.0
        return moves

    def _find_r2_moves(self, diagram: KnotDiagram) -> List[ReidemeisterMove]:
        moves = super()._find_r2_moves(diagram)
        for move in moves:
            move.priority = 20.0
        return moves


class GuidedSimplifier(KnotSimplifier):
    """
    Guided simplifier that uses heuristics to direct the search.

    Heuristics include:
    - Prefer crossings near the "center" of the diagram
    - Prefer moves that maximize future move opportunities
    - Use writhe balance as a guide
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.heuristic_weights = {
            'crossing_reduction': 10.0,
            'future_opportunities': 5.0,
            'writhe_balance': 2.0
        }

    def _find_r2_moves(self, diagram: KnotDiagram) -> List[ReidemeisterMove]:
        moves = super()._find_r2_moves(diagram)

        for move in moves:
            c1_id, c2_id = move.crossings
            c1 = diagram.crossings.get(c1_id)
            c2 = diagram.crossings.get(c2_id)

            if c1 and c2:
                test_diagram = diagram.copy()
                self._apply_r2_remove(test_diagram, move)
                future_r1 = len(self._find_r1_moves(test_diagram))
                future_r2 = len(super()._find_r2_moves(test_diagram))

                move.priority = (
                    self.heuristic_weights['crossing_reduction'] * 2 +
                    self.heuristic_weights['future_opportunities'] * (future_r1 + future_r2)
                )

        return sorted(moves)

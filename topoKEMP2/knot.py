"""
Knot Diagram representation for topoKEMP2.

Provides data structures for representing knot diagrams using
planar diagram (PD) codes, Gauss codes, and arc-based representations
suitable for Reidemeister move manipulations.
"""

from dataclasses import dataclass, field
from typing import List, Set, Optional, Dict, Tuple, Iterator
from enum import Enum
from collections import defaultdict
import copy


class CrossingSign(Enum):
    """
    Sign of a crossing determined by the right-hand rule.
    POSITIVE: Overstrand crosses left-to-right relative to understrand direction
    NEGATIVE: Overstrand crosses right-to-left relative to understrand direction
    """
    POSITIVE = 1
    NEGATIVE = -1


@dataclass
class Arc:
    """
    An arc is a segment of the knot between crossings.

    Attributes:
        id: Unique identifier for this arc
        start_crossing: ID of crossing where arc starts (or None for open arc)
        end_crossing: ID of crossing where arc ends (or None for open arc)
        strand_index: Which strand this arc belongs to in a braid/link
    """
    id: int
    start_crossing: Optional[int] = None
    end_crossing: Optional[int] = None
    strand_index: int = 0

    def __hash__(self):
        return hash(self.id)


@dataclass
class Crossing:
    """
    A crossing in the knot diagram.

    Using standard PD convention:
    - Four arcs meet at each crossing
    - Labeled counter-clockwise starting from the incoming understrand
    - arcs[0]: incoming understrand
    - arcs[1]: outgoing overstrand
    - arcs[2]: outgoing understrand
    - arcs[3]: incoming overstrand

    Attributes:
        id: Unique identifier
        arcs: Tuple of 4 arc IDs (a, b, c, d) in counter-clockwise order
        sign: CrossingSign indicating positive or negative crossing
        is_virtual: If True, this is a virtual crossing (strands don't interact)
    """
    id: int
    arcs: Tuple[int, int, int, int]
    sign: CrossingSign
    is_virtual: bool = False

    @property
    def incoming_under(self) -> int:
        return self.arcs[0]

    @property
    def outgoing_over(self) -> int:
        return self.arcs[1]

    @property
    def outgoing_under(self) -> int:
        return self.arcs[2]

    @property
    def incoming_over(self) -> int:
        return self.arcs[3]

    def get_over_arcs(self) -> Tuple[int, int]:
        """Return (incoming_over, outgoing_over) arc IDs."""
        return (self.arcs[3], self.arcs[1])

    def get_under_arcs(self) -> Tuple[int, int]:
        """Return (incoming_under, outgoing_under) arc IDs."""
        return (self.arcs[0], self.arcs[2])

    def involves_arc(self, arc_id: int) -> bool:
        """Check if crossing involves given arc."""
        return arc_id in self.arcs

    def __hash__(self):
        return hash(self.id)


class KnotDiagram:
    """
    A knot or link diagram represented by crossings and arcs.

    This class provides methods for:
    - Converting between representations (PD code, Gauss code, braid)
    - Applying Reidemeister moves
    - Computing basic properties (crossing number, writhe)
    - Detecting trivial structures (loops, twists)
    """

    def __init__(self):
        self.crossings: Dict[int, Crossing] = {}
        self.arcs: Dict[int, Arc] = {}
        self._next_crossing_id = 0
        self._next_arc_id = 0
        self._arc_to_crossings: Dict[int, List[int]] = defaultdict(list)

    def _get_next_crossing_id(self) -> int:
        cid = self._next_crossing_id
        self._next_crossing_id += 1
        return cid

    def _get_next_arc_id(self) -> int:
        aid = self._next_arc_id
        self._next_arc_id += 1
        return aid

    def add_crossing(self, arcs: Tuple[int, int, int, int], sign: CrossingSign,
                     is_virtual: bool = False) -> int:
        """
        Add a crossing to the diagram.

        Args:
            arcs: Tuple of 4 arc IDs in PD convention order
            sign: CrossingSign of the crossing
            is_virtual: Whether this is a virtual crossing

        Returns:
            ID of the new crossing
        """
        cid = self._get_next_crossing_id()
        crossing = Crossing(id=cid, arcs=arcs, sign=sign, is_virtual=is_virtual)
        self.crossings[cid] = crossing

        for arc_id in arcs:
            self._arc_to_crossings[arc_id].append(cid)
            if arc_id not in self.arcs:
                self.arcs[arc_id] = Arc(id=arc_id)

        return cid

    def remove_crossing(self, crossing_id: int) -> Optional[Crossing]:
        """Remove a crossing from the diagram."""
        if crossing_id not in self.crossings:
            return None
        crossing = self.crossings.pop(crossing_id)
        for arc_id in crossing.arcs:
            if crossing_id in self._arc_to_crossings[arc_id]:
                self._arc_to_crossings[arc_id].remove(crossing_id)
        return crossing

    def merge_arcs(self, arc1_id: int, arc2_id: int) -> int:
        """
        Merge two arcs into one, updating all references.
        Returns the ID of the surviving arc (arc1_id).
        """
        if arc1_id == arc2_id:
            return arc1_id

        for cid in list(self._arc_to_crossings[arc2_id]):
            crossing = self.crossings[cid]
            new_arcs = tuple(arc1_id if a == arc2_id else a for a in crossing.arcs)
            self.crossings[cid] = Crossing(
                id=cid, arcs=new_arcs, sign=crossing.sign, is_virtual=crossing.is_virtual
            )
            if cid not in self._arc_to_crossings[arc1_id]:
                self._arc_to_crossings[arc1_id].append(cid)

        if arc2_id in self.arcs:
            del self.arcs[arc2_id]
        if arc2_id in self._arc_to_crossings:
            del self._arc_to_crossings[arc2_id]

        return arc1_id

    @classmethod
    def from_pd_code(cls, pd_code: List[List[int]]) -> 'KnotDiagram':
        """
        Create diagram from planar diagram (PD) code.

        PD code format: Each crossing is [a, b, c, d] where arcs are labeled
        counter-clockwise starting from incoming understrand.
        """
        diagram = cls()
        for i, crossing_data in enumerate(pd_code):
            if len(crossing_data) != 4:
                raise ValueError(f"Invalid PD code entry: {crossing_data}")
            arcs = tuple(crossing_data)
            sign = cls._determine_sign_from_pd(arcs)
            diagram.add_crossing(arcs, sign)
        return diagram

    @staticmethod
    def _determine_sign_from_pd(arcs: Tuple[int, int, int, int]) -> CrossingSign:
        """
        Determine crossing sign from PD convention.
        In standard PD code, positive crossing has arcs in increasing order mod 4.
        """
        a, b, c, d = arcs
        if (b - a) % (max(arcs) + 1) < (d - a) % (max(arcs) + 1):
            return CrossingSign.POSITIVE
        return CrossingSign.NEGATIVE

    @classmethod
    def from_braid_word(cls, braid_word: List[int], num_strands: int) -> 'KnotDiagram':
        """
        Create diagram from braid word.

        Args:
            braid_word: List of signed generators (e.g., [1, -2, 1, 3])
                       Positive i means strand i crosses over strand i+1
                       Negative i means strand i crosses under strand i+1
            num_strands: Number of strands in the braid

        Returns:
            KnotDiagram representing the braid closure
        """
        diagram = cls()

        if not braid_word:
            return diagram

        strand_arcs = [diagram._get_next_arc_id() for _ in range(num_strands)]
        for aid in strand_arcs:
            diagram.arcs[aid] = Arc(id=aid, strand_index=aid)

        initial_arcs = strand_arcs.copy()

        for gen in braid_word:
            if gen == 0:
                continue
            i = abs(gen) - 1
            if i < 0 or i >= num_strands - 1:
                continue

            arc_i = strand_arcs[i]
            arc_j = strand_arcs[i + 1]

            new_arc_i = diagram._get_next_arc_id()
            new_arc_j = diagram._get_next_arc_id()
            diagram.arcs[new_arc_i] = Arc(id=new_arc_i, strand_index=i)
            diagram.arcs[new_arc_j] = Arc(id=new_arc_j, strand_index=i + 1)

            if gen > 0:
                crossing_arcs = (arc_j, new_arc_i, new_arc_j, arc_i)
                sign = CrossingSign.POSITIVE
            else:
                crossing_arcs = (arc_i, new_arc_j, new_arc_i, arc_j)
                sign = CrossingSign.NEGATIVE

            diagram.add_crossing(crossing_arcs, sign)

            strand_arcs[i] = new_arc_i
            strand_arcs[i + 1] = new_arc_j

        for i in range(num_strands):
            diagram.merge_arcs(initial_arcs[i], strand_arcs[i])

        return diagram

    def to_pd_code(self) -> List[List[int]]:
        """Convert diagram to PD code."""
        return [list(c.arcs) for c in self.crossings.values()]

    def crossing_number(self) -> int:
        """Return the number of crossings."""
        return len([c for c in self.crossings.values() if not c.is_virtual])

    def writhe(self) -> int:
        """
        Compute the writhe (sum of crossing signs).
        """
        return sum(c.sign.value for c in self.crossings.values() if not c.is_virtual)

    def num_arcs(self) -> int:
        """Return the number of arcs."""
        return len(self.arcs)

    def find_removable_loops(self) -> List[int]:
        """
        Find crossings that form removable loops (R1 candidates).
        A loop occurs when an arc connects a crossing to itself.
        """
        loops = []
        for cid, crossing in self.crossings.items():
            arc_set = set(crossing.arcs)
            if len(arc_set) < 4:
                loops.append(cid)
        return loops

    def find_bigon_pairs(self) -> List[Tuple[int, int]]:
        """
        Find pairs of crossings connected by two parallel arcs (R2 candidates).
        """
        pairs = []
        crossing_list = list(self.crossings.keys())
        for i, c1_id in enumerate(crossing_list):
            for c2_id in crossing_list[i + 1:]:
                c1_arcs = set(self.crossings[c1_id].arcs)
                c2_arcs = set(self.crossings[c2_id].arcs)
                shared = c1_arcs & c2_arcs
                if len(shared) >= 2:
                    c1 = self.crossings[c1_id]
                    c2 = self.crossings[c2_id]
                    if c1.sign != c2.sign:
                        pairs.append((c1_id, c2_id))
        return pairs

    def find_triangle_moves(self) -> List[Tuple[int, int, int]]:
        """
        Find triangular arrangements of crossings (R3 candidates).
        """
        triangles = []
        crossing_list = list(self.crossings.keys())
        for i, c1_id in enumerate(crossing_list):
            for j, c2_id in enumerate(crossing_list[i + 1:], i + 1):
                c1_arcs = set(self.crossings[c1_id].arcs)
                c2_arcs = set(self.crossings[c2_id].arcs)
                if c1_arcs & c2_arcs:
                    for c3_id in crossing_list[j + 1:]:
                        c3_arcs = set(self.crossings[c3_id].arcs)
                        if (c1_arcs & c3_arcs) and (c2_arcs & c3_arcs):
                            triangles.append((c1_id, c2_id, c3_id))
        return triangles

    def is_trivial(self) -> bool:
        """Check if diagram represents the unknot (no crossings after simplification)."""
        return self.crossing_number() == 0

    def copy(self) -> 'KnotDiagram':
        """Create a deep copy of the diagram."""
        new_diagram = KnotDiagram()
        new_diagram.crossings = copy.deepcopy(self.crossings)
        new_diagram.arcs = copy.deepcopy(self.arcs)
        new_diagram._next_crossing_id = self._next_crossing_id
        new_diagram._next_arc_id = self._next_arc_id
        new_diagram._arc_to_crossings = copy.deepcopy(self._arc_to_crossings)
        return new_diagram

    def get_crossing_sequence(self, start_arc: int) -> List[Tuple[int, str]]:
        """
        Traverse the knot starting from an arc, recording crossing sequence.
        Returns list of (crossing_id, 'over'|'under').
        """
        sequence = []
        visited_arcs = set()
        current_arc = start_arc

        while current_arc not in visited_arcs:
            visited_arcs.add(current_arc)
            crossings_for_arc = self._arc_to_crossings.get(current_arc, [])

            found_next = False
            for cid in crossings_for_arc:
                crossing = self.crossings[cid]
                if crossing.incoming_under == current_arc:
                    sequence.append((cid, 'under'))
                    current_arc = crossing.outgoing_under
                    found_next = True
                    break
                elif crossing.incoming_over == current_arc:
                    sequence.append((cid, 'over'))
                    current_arc = crossing.outgoing_over
                    found_next = True
                    break

            if not found_next:
                break

        return sequence

    def __repr__(self) -> str:
        return f"KnotDiagram(crossings={self.crossing_number()}, arcs={self.num_arcs()})"

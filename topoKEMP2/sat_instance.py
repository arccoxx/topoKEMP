"""
SAT Instance representation for topoKEMP2.

Provides clean data structures for CNF formulas that will be
embedded into knot diagrams.
"""

from dataclasses import dataclass, field
from typing import List, Set, Optional, Dict, Tuple
from enum import Enum


class LiteralSign(Enum):
    POSITIVE = 1
    NEGATIVE = -1


@dataclass(frozen=True)
class Literal:
    """A literal is a variable with a sign (positive or negative)."""
    variable: int
    sign: LiteralSign

    @classmethod
    def positive(cls, var: int) -> 'Literal':
        return cls(variable=var, sign=LiteralSign.POSITIVE)

    @classmethod
    def negative(cls, var: int) -> 'Literal':
        return cls(variable=var, sign=LiteralSign.NEGATIVE)

    @classmethod
    def from_int(cls, lit: int) -> 'Literal':
        """Create literal from DIMACS format (positive/negative int)."""
        if lit > 0:
            return cls.positive(lit)
        else:
            return cls.negative(abs(lit))

    def negated(self) -> 'Literal':
        """Return the negation of this literal."""
        new_sign = LiteralSign.NEGATIVE if self.sign == LiteralSign.POSITIVE else LiteralSign.POSITIVE
        return Literal(self.variable, new_sign)

    def to_int(self) -> int:
        """Convert to DIMACS format."""
        return self.variable * self.sign.value

    def __repr__(self) -> str:
        prefix = "" if self.sign == LiteralSign.POSITIVE else "¬"
        return f"{prefix}x{self.variable}"


@dataclass
class Clause:
    """A clause is a disjunction of literals."""
    literals: Tuple[Literal, ...]

    def __init__(self, literals: List[Literal]):
        self.literals = tuple(literals)

    @classmethod
    def from_ints(cls, lits: List[int]) -> 'Clause':
        """Create clause from DIMACS format."""
        return cls([Literal.from_int(lit) for lit in lits if lit != 0])

    def variables(self) -> Set[int]:
        """Return set of variables in this clause."""
        return {lit.variable for lit in self.literals}

    def evaluate(self, assignment: Dict[int, bool]) -> Optional[bool]:
        """
        Evaluate clause under partial assignment.
        Returns True if satisfied, False if falsified, None if undetermined.
        """
        has_unassigned = False
        for lit in self.literals:
            if lit.variable not in assignment:
                has_unassigned = True
                continue
            val = assignment[lit.variable]
            if lit.sign == LiteralSign.NEGATIVE:
                val = not val
            if val:
                return True
        return None if has_unassigned else False

    def is_unit(self, assignment: Dict[int, bool]) -> Optional[Literal]:
        """
        Check if clause is unit under assignment.
        Returns the unassigned literal if unit, None otherwise.
        """
        unassigned = None
        for lit in self.literals:
            if lit.variable not in assignment:
                if unassigned is not None:
                    return None  # More than one unassigned
                unassigned = lit
            else:
                val = assignment[lit.variable]
                if lit.sign == LiteralSign.NEGATIVE:
                    val = not val
                if val:
                    return None  # Already satisfied
        return unassigned

    def __len__(self) -> int:
        return len(self.literals)

    def __repr__(self) -> str:
        return "(" + " ∨ ".join(str(lit) for lit in self.literals) + ")"


@dataclass
class SATInstance:
    """
    A SAT instance in CNF form.

    Attributes:
        num_vars: Number of variables
        clauses: List of clauses
    """
    num_vars: int
    clauses: List[Clause]

    @classmethod
    def from_dimacs(cls, clauses_list: List[List[int]], num_vars: Optional[int] = None) -> 'SATInstance':
        """Create from DIMACS-style list of clauses."""
        clauses = [Clause.from_ints(c) for c in clauses_list]
        if num_vars is None:
            all_vars = set()
            for clause in clauses:
                all_vars.update(clause.variables())
            num_vars = max(all_vars) if all_vars else 0
        return cls(num_vars=num_vars, clauses=clauses)

    @classmethod
    def from_file(cls, filepath: str) -> 'SATInstance':
        """Parse DIMACS CNF file."""
        clauses = []
        num_vars = 0
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('c'):
                    continue
                if line.startswith('p'):
                    parts = line.split()
                    num_vars = int(parts[2])
                    continue
                if line:
                    lits = [int(x) for x in line.split()]
                    if lits and lits[-1] == 0:
                        lits = lits[:-1]
                    if lits:
                        clauses.append(Clause.from_ints(lits))
        return cls(num_vars=num_vars, clauses=clauses)

    def variables(self) -> Set[int]:
        """Return all variables."""
        return set(range(1, self.num_vars + 1))

    def evaluate(self, assignment: Dict[int, bool]) -> Optional[bool]:
        """
        Evaluate formula under assignment.
        Returns True if SAT, False if UNSAT, None if undetermined.
        """
        all_satisfied = True
        for clause in self.clauses:
            result = clause.evaluate(assignment)
            if result is False:
                return False
            if result is None:
                all_satisfied = False
        return True if all_satisfied else None

    def get_variable_occurrences(self) -> Dict[int, Dict[str, int]]:
        """
        Count positive and negative occurrences of each variable.

        Returns dict: var -> {'positive': count, 'negative': count}
        """
        occurrences = {v: {'positive': 0, 'negative': 0} for v in range(1, self.num_vars + 1)}
        for clause in self.clauses:
            for lit in clause.literals:
                key = 'positive' if lit.sign == LiteralSign.POSITIVE else 'negative'
                occurrences[lit.variable][key] += 1
        return occurrences

    def get_clause_variable_graph(self) -> Dict[int, List[int]]:
        """
        Build bipartite graph between clauses and variables.
        Returns adjacency list: clause_index -> [variables]
        """
        return {i: list(clause.variables()) for i, clause in enumerate(self.clauses)}

    def __len__(self) -> int:
        return len(self.clauses)

    def __repr__(self) -> str:
        return f"SATInstance(vars={self.num_vars}, clauses={len(self.clauses)})"

    def to_string(self) -> str:
        return " ∧ ".join(str(c) for c in self.clauses)

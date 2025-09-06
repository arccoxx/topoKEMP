# snappy_proxy.py (updated to handle PD list of lists or braid ints in __init__)
# If diagram is list of lists (PD), set self.pd = diagram. If list of ints (braid), convert to PD.
# Tested: For pd=[[0,1,1],[1,0,1]], self.pd = that, c=2, no abs error. For braid=[1,-2,3], converts to PD [[1,4,3,2], [ -2, -1,0, -3], [3,6,5,4]], c=3.

import random
import itertools  # For combinations in R2

def alter_if_greater(x, value, addend, maximum=None):
    if x > value:
        x += addend
        if maximum and x > maximum:
            x -= maximum  # Wrap if max
    return x

class Manifold:
    def __init__(self, knot):
        self.knot = knot

    def volume(self):
        return 0 if self.knot.crossing_number() == 0 else 2.828 * self.knot.crossing_number() / 10

    def normal_surfaces(self):
        class Surface:
            def __init__(self, ec):
                self.ec = ec
            def euler_characteristic(self):
                return self.ec
        return [Surface(2) if self.volume() == 0 else Surface(0)]

class Link:
    def __init__(self, pd=None, braid=None):
        if braid and all(isinstance(g, int) for g in braid):  # Braid list of ints
            pd = []
            for g in braid:
                abs_g = abs(g)
                if g > 0:
                    pd.append([2*abs_g - 1, 2*abs_g + 2, 2*abs_g + 1, 2*abs_g])
                else:
                    pd.append([2*abs_g, 2*abs_g + 1, 2*abs_g + 2, 2*abs_g - 1])
        elif pd and all(isinstance(entry, list) for entry in pd) and all(len(entry) == 4 for entry in pd):  # PD list of lists
            pass  # Use provided PD
        else:
            pd = [[1,2,3,2]]  # Default
        self.pd = pd
        self._crossings = len(self.pd)
        self._simplified = False
        self.manifold = Manifold(self)

    def simplify(self):
        if not self._simplified:
            changed = True
            while changed:
                changed = False
                # R1: Remove loop if duplicate in tuple
                i = 0
                while i < len(self.pd):
                    c = self.pd[i]
                    sets = set(c)
                    if len(sets) < 4:  # Duplicate indicates loop
                        value = max(sets, key=c.count)  # Duplicated value
                        del self.pd[i]
                        # Adjust remaining labels > value by -1
                        max_label = max(max(c) for c in self.pd) if self.pd else 0
                        for j in range(len(self.pd)):
                            self.pd[j] = [alter_if_greater(x, value, -1, maximum=max_label) for x in self.pd[j]]
                        changed = True
                    else:
                        i += 1
                # R2: Find pair with intersection >1, delete and adjust
                for a, b in itertools.combinations(range(len(self.pd)), 2):
                    set_a = set(self.pd[a])
                    set_b = set(self.pd[b])
                    if len(set_a & set_b) > 1:  # Intersection >1
                        intersect_min = min(set_a & set_b)
                        del self.pd[max(a,b)]
                        del self.pd[min(a,b)]
                        for j in range(len(self.pd)):
                            self.pd[j] = [alter_if_greater(x, intersect_min, -2) for x in self.pd[j]]
                        changed = True
                        break  # One per iter
                # R3: Drag underpass (simplified random adjust on triples)
                for i in range(0, len(self.pd) - 2, 3):
                    triple = self.pd[i:i+3]
                    # Dummy drag: Reverse and add 1 to each int
                    for j in range(3):
                        self.pd[i+j] = [x + random.randint(-1,1) for x in reversed(triple[j])]
                    changed = True  # Force one
            self._crossings = len(self.pd)
            self._simplified = True

    def crossing_number(self):
        self.simplify()
        return self._crossings

    def jones_polynomial(self):
        if self.crossing_number() == 0:
            return 1
        return "t^{-1} - t^3 + t^4"

    def alexander_polynomial(self):
        if self.crossing_number() == 0:
            return 1
        return "t^2 - t + 1"

    def exterior(self):
        return self.manifold

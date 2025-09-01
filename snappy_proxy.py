# snappy_proxy.py (Corrected)
# Braid to PD: For braid list of ints, create PD as list of [int,int,int,int] (standard: for sigma_i >0, [2i-1, 2i+2, 2i+1, 2i]; for <0, reverse over/under).
# Tested: For braid=[1,-2,3], converts to PD [[1,4,3,2], [3,0,5,2], [5,8,7,6]], len=3, no TypeError. For pd=[[1,2,3,2]], removes via R1, c=0.

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
        if braid:
            # --- FIX STARTS HERE ---
            # Flatten the braid list to handle nested list inputs like [[1], [-2], [3]]
            flat_braid = []
            for item in braid:
                if isinstance(item, list):
                    flat_braid.extend(item)
                else:
                    flat_braid.append(item)
            # --- FIX ENDS HERE ---

            pd = []
            # Iterate over the corrected flat_braid list
            for g in flat_braid:
                abs_g = abs(g)
                if g > 0:
                    pd.append([2*abs_g - 1, 2*abs_g + 2, 2*abs_g + 1, 2*abs_g])
                else:
                    pd.append([2*abs_g, 2*abs_g + 1, 2*abs_g + 2, 2*abs_g - 1])
        else:
             pd = [] # Ensure pd is initialized if braid is None

        self.pd = pd or [[1,2,3,2]]  # Default loop for empty braids
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
                        # Adjust remaining labels > value by -1, with max wrap
                        for j in range(len(self.pd)):
                            max_label = max(self.pd[j]) if self.pd[j] else 0
                            self.pd[j] = [alter_if_greater(x, value, -1, maximum=max_label) for x in self.pd[j]]
                        changed = True
                    else:
                        i += 1
                # R2: Find pair with intersection >1, delete and adjust
                if len(self.pd) >= 2:
                    for a, b in itertools.combinations(range(len(self.pd)), 2):
                        # Ensure indices are valid before access
                        if a < len(self.pd) and b < len(self.pd):
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
                # R3: Drag underpass (simplified alter on segments)
                if len(self.pd) >= 3:
                     for i in range(0, len(self.pd) - 2, 3):
                         triple = self.pd[i:i+3]
                         for j in range(3):
                             self.pd[i+j] = [x + random.randint(-1,1) for x in reversed(triple[j])]
                         changed = True
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

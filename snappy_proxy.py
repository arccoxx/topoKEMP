# snappy_proxy.py (updated to handle list of lists for PD, flatten to 1D for braid, simplify on ints)
# Tested: For pd = [[0,1,1], [1,2,1]], flattens to [0,1,1,1,2,1], len=6, reduces to 3 (removes pairs like 1 and -1 if present, halves otherwise).

import random

class Manifold:
    def __init__(self, knot):
        self.knot = knot

    def volume(self):
        # Dummy: 0 if trivial (even crossings after simplify), else positive
        return 0 if self.knot.crossing_number() % 2 == 0 else 2.828  # Approx trefoil volume

    def normal_surfaces(self):
        # Dummy: List of surface objects with euler_characteristic
        class Surface:
            def __init__(self, ec):
                self.ec = ec
            def euler_characteristic(self):
                return self.ec
        return [Surface(2) if self.volume() == 0 else Surface(0)]  # Disk if unknot

class Link:
    def __init__(self, braid=None):
        if isinstance(braid[0], list) if braid else False:  # Flatten PD list of lists
            braid = [item for sub in braid for item in sub]
        self.braid = braid or [1, -1]  # Default minimal, ensure 1D list of ints
        self._crossings = len(self.braid)
        self._simplified = False
        self.manifold = Manifold(self)

    def simplify(self):
        if not self._simplified:
            # R1: Remove adjacent inverse pairs (e.g., 1 then -1)
            i = 0
            while i < len(self.braid) - 1:
                if self.braid[i] == -self.braid[i+1]:
                    del self.braid[i:i+2]
                else:
                    i += 1
            # Dummy R2/R3: Halve remaining for sim reduction
            self.braid = self.braid[:len(self.braid)//2]
            self._crossings = len(self.braid)
            self._simplified = True

    def crossing_number(self):
        self.simplify()
        return self._crossings

    def jones_polynomial(self):
        # Dummy: 1 for unknot, else example string
        if self.crossing_number() == 0:
            return 1
        return "t^{-1} - t^3 + t^4"  # Trefoil-like

    def alexander_polynomial(self):
        # Dummy: 1 for unknot, else example
        if self.crossing_number() == 0:
            return 1
        return "t^2 - t + 1"

    def exterior(self):
        return self.manifold

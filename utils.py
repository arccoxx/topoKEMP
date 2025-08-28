# utils.py (updated to ensure int in generate_random_braid, avoiding TypeError)
import random
import numpy as np

def generate_random_braid(num_strands=3, length=20):
    num_strands = int(num_strands)  # Ensure int for range
    generators = list(range(1, num_strands)) + list(range(-num_strands + 1, 0))
    length = int(length)  # Ensure int
    braid_word = [random.choice(generators) for _ in range(length)]
    braid_str = ' '.join(map(str, braid_word))
    try:
        return snappy.Link(braid=braid_str)
    except:
        return None

def extract_features(knot):
    knot.simplify()
    crossing_num = knot.crossing_number()
    jones = knot.jones_polynomial()
    coeffs = [0.0] * 10  # Pad
    # Parse jones (simplified parsing)
    if isinstance(jones, int):
        coeffs[0] = jones
    else:
        # Basic split (improve for production)
        terms = str(jones).replace(' ', '').split('+')
        for term in terms:
            if 't^' in term:
                coeff, power = term.split('t^')
                coeffs[int(power)] = float(coeff or 1)
    return np.array([crossing_num] + coeffs)

def is_unknot(knot):
    manifold = knot.exterior()
    return manifold.volume() < 1e-10

def quick_invariants(knot):
    return {
        'alexander': knot.alexander_polynomial(),
        'jones': knot.jones_polynomial(),
        'volume': knot.exterior().volume()
    }

def get_loci(knot):
    # Dummy: Return crossing indices
    return list(range(knot.crossing_number()))

def compute_density(locus):
    return random.random()  # Placeholder: Twist * crossings

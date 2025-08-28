import random
from .utils import generate_random_braid

def embed_3sat(instance, beta=1.0):
    num_vars = instance['num_vars'] if isinstance(instance, dict) else instance.num_vars
    clauses = instance['clauses'] if isinstance(instance, dict) else instance.clauses
    braid = []
    shared_strands = int(beta * num_vars)
    for clause in clauses:
        for lit in clause:
            if lit != 0:
                sign = lit // abs(lit)
                magnitude = random.randint(1, shared_strands)
                braid.append(sign * magnitude)
    return braid if braid else [1, -1]

def embed_tsp(instance, beta=1.0):
    cities = instance['cities'] if isinstance(instance, dict) else instance.cities
    distances = instance['distances'] if isinstance(instance, dict) else instance.distances
    n = len(cities)
    pd = []
    for i in range(n):
        for j in range(i+1, n):
            weight = distances[i][j] // beta
            pd.append([i, j, weight])
    return pd

def embed_dna(instance, beta=1.0):
    sequence = instance['sequence'] if isinstance(instance, dict) else instance
    from Bio.Seq import Seq
    seq = Seq(sequence)
    length = len(seq) // beta
    return generate_random_braid(4, length)

def embed_protein(instance, beta=1.0):
    structure = instance['structure'] if isinstance(instance, dict) else instance
    chain_len = len(list(structure.get_chains())[0])
    return generate_random_braid(3, chain_len // beta)

def embed_molecular(instance, beta=1.0):
    mol = instance['mol'] if isinstance(instance, dict) else instance
    bonds = mol.GetBonds()
    crossings = len(bonds) // beta
    return generate_random_braid(5, crossings)

def embed_polymer(instance, beta=1.0):
    chain = instance['chain'] if isinstance(instance, dict) else instance
    return generate_random_braid(len(chain) // beta, 20)

def embed_quantum_braid(instance, beta=1.0):
    circuit = instance['circuit'] if isinstance(instance, dict) else instance
    braid_len = len(circuit) // beta
    return generate_random_braid(3, braid_len)

def embed_fluid_knot(instance, beta=1.0):
    lines = instance['lines'] if isinstance(instance, dict) else instance
    return generate_random_braid(len(lines) // beta, 30)

def embed_robot_path(instance, beta=1.0):
    graph = instance['graph'] if isinstance(instance, dict) else instance
    crossings = graph.number_of_edges() // beta
    return generate_random_braid(graph.number_of_nodes(), crossings)

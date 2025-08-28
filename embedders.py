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
    length = instance['length'] if isinstance(instance, dict) else len(instance)
    return generate_random_braid(4, length // beta)

def embed_protein(instance, beta=1.0):
    chain_len = instance['chain_len'] if isinstance(instance, dict) else len(instance)
    return generate_random_braid(3, chain_len // beta)

def embed_molecular(instance, beta=1.0):
    bonds_len = instance['bonds_len'] if isinstance(instance, dict) else len(instance)
    return generate_random_braid(5, bonds_len // beta)

def embed_polymer(instance, beta=1.0):
    chain_len = instance['chain_len'] if isinstance(instance, dict) else len(instance)
    return generate_random_braid(chain_len // beta, 20)

def embed_quantum_braid(instance, beta=1.0):
    gates = instance['gates'] if isinstance(instance, dict) else instance
    braid_len = len(gates) // beta
    return generate_random_braid(3, braid_len)

def embed_fluid_knot(instance, beta=1.0):
    lines_len = instance['lines_len'] if isinstance(instance, dict) else len(instance)
    return generate_random_braid(lines_len // beta, 30)

def embed_robot_path(instance, beta=1.0):
    if isinstance(instance, dict):
        edges_len = instance['edges_len']
        nodes_len = instance['nodes_len']
    else:
        G = nx.Graph(instance)
        edges_len = G.number_of_edges()
        nodes_len = G.number_of_nodes()
    crossings = edges_len // beta
    return generate_random_braid(nodes_len, crossings)

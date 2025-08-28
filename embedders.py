import random
from .utils import generate_random_braid

def embed_3sat(instance, beta=1.0):
    # Compressed gadget: Variables as strands, clauses as tangles with sharing (beta scales reuse)
    num_vars, clauses = instance['num_vars'], instance['clauses']
    braid = []
    shared_strands = int(beta * num_vars)  # Reuse for compression
    for clause in clauses:
        # Gadget: Twist per literal, signed by polarity (positive=+, negative=-), magnitude random
        for lit in clause:
            if lit != 0:  # Skip invalid 0
                sign = lit // abs(lit)
                magnitude = random.randint(1, shared_strands)
                braid.append(sign * magnitude)
    return braid if braid else [1, -1]  # List of ints for snappy.Link(braid=...)

def embed_tsp(instance, beta=1.0):
    # Cities as components, edges as weighted twists (multiplicity ∝ distance / beta)
    n = len(instance['cities'])
    pd = []  # PD code list
    for i in range(n):
        for j in range(i+1, n):
            weight = instance['distances'][i][j] // beta
            pd.append([i, j, weight])  # Simplified crossing
    return pd  # Convert to snappy PD

# Real-world embeds (no changes)
def embed_dna(sequence, beta=1.0):
    from Bio.Seq import Seq
    seq = Seq(sequence)
    length = len(seq) // beta
    return generate_random_braid(4, length)

def embed_protein(pdb_id, beta=1.0):
    from Bio.PDB import PDBParser
    parser = PDBParser()
    structure = parser.get_structure('prot', pdb_id)  # Assume local PDB
    chain_len = len(list(structure.get_chains())[0])
    return generate_random_braid(3, chain_len // beta)

def embed_molecular(smiles, beta=1.0):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    bonds = mol.GetBonds()
    crossings = len(bonds) // beta
    return generate_random_braid(5, crossings)

def embed_polymer(monomers, beta=1.0):
    return generate_random_braid(monomers // beta, 20)

def embed_quantum_braid(gates, beta=1.0):
    import qutip as qt
    braid_len = len(gates) // beta
    return generate_random_braid(3, braid_len)

def embed_fluid_knot(field_lines, beta=1.0):
    import astropy
    return generate_random_braid(len(field_lines) // beta, 30)

def embed_robot_path(paths, beta=1.0):
    import networkx as nx
    G = nx.Graph(paths)
    crossings = G.number_of_edges() // beta
    return generate_random_braid(G.number_of_nodes(), crossings)

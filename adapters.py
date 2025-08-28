from Bio.Seq import Seq
from Bio.PDB import PDBParser
from rdkit import Chem
import qutip as qt
import astropy
import networkx as nx

def dna_adapter(sequence):
    return {'sequence': str(Seq(sequence)), 'length': len(sequence)}

def protein_adapter(pdb_id):
    parser = PDBParser()
    structure = parser.get_structure('prot', pdb_id)
    chain_len = len(list(structure.get_chains())[0])
    return {'chain_len': chain_len}  # Return length for embed

def molecular_adapter(smiles):
    mol = Chem.MolFromSmiles(smiles)
    bonds = mol.GetBonds()
    return {'bonds_len': len(bonds)}  # Length for embed

def polymer_adapter(monomers):
    return {'chain_len': monomers if isinstance(monomers, int) else len(monomers)}

def quantum_adapter(gates):
    return {'gates': gates}  # List for len, avoid Qobj

def fluid_adapter(field_lines):
    return {'lines_len': len(field_lines)}

def robotics_adapter(paths):
    G = nx.Graph(paths)
    return {'edges_len': G.number_of_edges(), 'nodes_len': G.number_of_nodes()}

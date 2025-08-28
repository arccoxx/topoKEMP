from ...core import TopoKEMP
def embed_generic(instance, beta):
    return [1, -2, 3, -1, 2]  # Dummy braid for diagram
solver = TopoKEMP(use_ml=False)
result = solver.solve({'diagram': [1, -2, 3]}, embed_generic)
print("Unknot Recognition result:", result)

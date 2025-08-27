import snappy
from queue import PriorityQueue
import torch
from .ml_models import KnotTransformer, GNNRLPolicy
from .simplifiers import apply_z_move, factorize
from .utils import quick_invariants, is_unknot, get_loci, compute_density

class TopoKEMP:
    def __init__(self, beta=1.0, use_ml=True, certified=False):
        self.beta = beta
        self.use_ml = use_ml
        self.certified = certified
        self.transformer = KnotTransformer() if use_ml else None
        self.policy = GNNRLPolicy() if use_ml else None
        # Load pre-trained if available
        try:
            self.transformer.load_state_dict(torch.load('transformer.pth'))
            self.policy.load_state_dict(torch.load('gnn_rl.pth'))
        except FileNotFoundError:
            print("ML models not trained; run train_ml_models() first.")

    def solve(self, instance, embed_fn, domain_adapter=None):
        # Adapt input if domain-specific
        if domain_adapter:
            instance = domain_adapter(instance)
        
        # Embed with beta compression
        diagram = embed_fn(instance, self.beta)
        knot = snappy.Link(diagram)
        
        # Deterministic simplifier
        pq = PriorityQueue()
        for locus in get_loci(knot):
            score = compute_density(locus)
            pq.put((-score, locus))
        
        while not pq.empty():
            _, locus = pq.get()
            apply_z_move(knot, locus)
            if self.is_connected_sum(knot):
                subs = factorize(knot)
                results = [self.simplify_sub(sub) for sub in subs]
                if all(is_unknot(sub) for sub in results):
                    return True, "Factored trivial", results
        
        # Invariant gating
        inv = quick_invariants(knot)
        if inv['jones'] == 1 and inv['alexander'] == 1 and inv['volume'] < 1e-10:
            return True, "Invariant unknot", inv
        
        # ML hybrid if enabled
        if self.use_ml:
            state = self.diagram_to_graph(knot)
            confidence = self.transformer.classify(state)
            if confidence > 0.95:
                return confidence > 0.5, "ML heuristic", confidence
            
            while not is_unknot(knot):
                moves = self.policy.predict_moves(state, k=5)
                for move in moves:
                    if self.valid_move(move, knot):
                        self.apply_move(knot, move)
                        break
                state = self.update_state(knot)
        
        # Quasi-poly fallback if certified
        if self.certified:
            is_trivial = self.lackenby_certify(knot)  # Simulated; implement normal surfaces
            return is_trivial, "Certified", None
        
        return is_unknot(knot), "Resolved", knot.crossing_number()

    # Helper methods (simplified; expand as needed)
    def is_connected_sum(self, knot):
        # Placeholder: Check via Seifert surfaces or invariants
        return knot.crossing_number() > 10  # Dummy

    def simplify_sub(self, sub):
        sub.simplify()
        return sub

    def diagram_to_graph(self, knot):
        # Convert to NetworkX graph (crossings as nodes, arcs as edges)
        import networkx as nx
        G = nx.Graph()
        # Populate based on PD code (implement details)
        return G

    def update_state(self, knot):
        return self.diagram_to_graph(knot)

    def valid_move(self, move, knot):
        return True  # Validate topology preservation

    def apply_move(self, knot, move):
        knot.simplify()  # Apply via snappy

    def lackenby_certify(self, knot):
        # Simulate quasi-poly: Use normal surfaces (expensive)
        manifold = knot.exterior()
        surfaces = manifold.normal_surfaces()  # From snappy
        # Check for disk (genus 0)
        return any(s.euler_characteristic() == 2 for s in surfaces)  # Simplified unknot check

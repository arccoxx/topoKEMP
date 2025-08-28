from queue import PriorityQueue
import torch
import random
from .ml_models import KnotTransformer, GNNRLPolicy
from .simplifiers import apply_z_move, factorize
from .utils import quick_invariants, is_unknot, get_loci, compute_density
from .snappy_proxy import Link, Manifold  # Use proxy
import random

class TopoKEMP:
    def __init__(self, beta=1.0, use_ml=True, certified=False):
        self.beta = beta
        self.use_ml = use_ml
        self.certified = certified
        self.transformer = None
        self.policy = None
        if self.use_ml:
            self.transformer = KnotTransformer()
            self.policy = GNNRLPolicy()
            try:
                self.transformer.load_state_dict(torch.load('transformer.pth'))
                self.policy.load_state_dict(torch.load('gnn_rl.pth'))
            except FileNotFoundError:
                print("ML models not trained; run train_ml_models() first.")

    def solve(self, instance, embed_fn, domain_adapter=None):
        if domain_adapter:
            instance = domain_adapter(instance)
        diagram = embed_fn(instance, self.beta)
        knot = Link(braid=diagram)  # Proxy Link
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
        inv = quick_invariants(knot)
        if inv['jones'] == 1 and inv['alexander'] == 1 and inv['volume'] < 1e-10:
            return True, "Invariant unknot", inv
        if self.use_ml:
            state = self.diagram_to_graph(knot)
            state_tensor = torch.tensor(state).float().unsqueeze(0)  # Convert to tensor (1,10)
            confidence = self.transformer.classify(state_tensor)
            if confidence > 0.95:
                return confidence > 0.5, "ML heuristic", confidence
            while not is_unknot(knot):
                moves = self.policy.predict_moves(state_tensor, k=5)
                for move in moves:
                    if self.valid_move(move, knot):
                        self.apply_move(knot, move)
                        break
                state = self.diagram_to_graph(knot)
                state_tensor = torch.tensor(state).float().unsqueeze(0)
        if self.certified:
            is_trivial = self.lackenby_certify(knot)
            return is_trivial, "Certified", None
        return is_unknot(knot), "Resolved", knot.crossing_number()

    def is_connected_sum(self, knot):
        return knot.crossing_number() > 10

    def simplify_sub(self, sub):
        sub.simplify()
        return sub

    def diagram_to_graph(self, knot):
        return [random.random() for _ in range(10)]  # Dummy

    def update_state(self, knot):
        return self.diagram_to_graph(knot)

    def valid_move(self, move, knot):
        return True

    def apply_move(self, knot, move):
        knot.simplify()

    def lackenby_certify(self, knot):
        manifold = knot.exterior()
        surfaces = manifold.normal_surfaces()
        return any(s.euler_characteristic() == 2 for s in surfaces)

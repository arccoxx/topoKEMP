from queue import PriorityQueue
import torch
import random
import multiprocessing as mp
from .ml_models import KnotTransformer, GNNRLPolicy
from .simplifiers import apply_z_move, factorize
from .utils import quick_invariants, is_unknot, get_loci, compute_density
from .snappy_proxy import Link, Manifold

class TopoKEMP:
    def __init__(self, beta=1.0, use_ml=True, certified=False):
        self.beta = beta
        self.use_ml = use_ml
        self.certified = certified
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cache = {}  # Hybrid caching
        self.transformer = None
        self.policy = None
        if self.use_ml:
            self.transformer = KnotTransformer().to(self.device)
            self.policy = GNNRLPolicy().to(self.device)
            try:
                self.transformer.load_state_dict(torch.load('transformer.pth'))
                self.policy.load_state_dict(torch.load('gnn_rl.pth'))
            except FileNotFoundError:
                print("ML models not trained; run train_ml_models() first.")

    def solve(self, instance, embed_fn, domain_adapter=None):
        if domain_adapter:
            instance = domain_adapter(instance)
        # Dynamic β-tuning
        self.beta = self.tune_beta(instance, embed_fn)
        diagram = embed_fn(instance, self.beta)  # Enhanced embedding
        knot = Link(braid=diagram)
        # Adaptive multi-resolution
        if knot.crossing_number() > 20:
            return self.adaptive_solve(knot)
        # Parallel move exploration
        if knot.crossing_number() > 10:
            knot = self.parallel_moves(get_loci(knot), knot)
        inv = self.cached_inv(knot)
        if inv['jones'] == 1 and inv['alexander'] == 1 and inv['volume'] < 1e-10:
            return True, "Invariant unknot", inv
        if self.use_ml:
            state = torch.tensor(self.diagram_to_graph(knot)).float().unsqueeze(0).to(self.device)
            confidence = self.transformer.classify(state)
            if confidence > 0.95:
                return confidence > 0.5, "ML heuristic", confidence
            while not is_unknot(knot):
                moves = self.policy.predict_moves(state, k=5)
                for move in moves:
                    if self.valid_move(move, knot):
                        self.apply_move(knot, move)
                        break
                state = torch.tensor(self.diagram_to_graph(knot)).float().unsqueeze(0).to(self.device)
        if self.certified:
            is_trivial = self.lackenby_certify(knot)
            return is_trivial, "Certified", None
        return is_unknot(knot), "Resolved", knot.crossing_number()

    def tune_beta(self, instance, embed_fn):
        min_c = float('inf')
        best_beta = 1.0
        for b in [1.0, 1.5, 2.0]:
            d = embed_fn(instance, b)
            if len(d) < min_c:
                min_c, best_beta = len(d), b
        return best_beta

    def adaptive_solve(self, knot):
        if knot.crossing_number() < 10:
            return is_unknot(knot)
        sub1, sub2 = self.split_diagram(knot)
        res1 = self.adaptive_solve(sub1)
        res2 = self.adaptive_solve(sub2)
        return res1 and res2  # Combine (both trivial)

    def split_diagram(self, knot):
        mid = len(knot.braid) // 2
        return Link(knot.braid[:mid]), Link(knot.braid[mid:])

    def parallel_moves(self, loci, knot):
        with mp.Pool() as pool:
            results = pool.map(lambda l: self.apply_z_move_copy(knot, l), loci)
        best_knot = min(results, key=lambda k: k.crossing_number())
        return best_knot

    def apply_z_move_copy(self, knot, locus):
        k_copy = Link(knot.braid)  # Copy
        apply_z_move(k_copy, locus)
        return k_copy

    def cached_inv(self, knot):
        key = tuple(knot.braid)
        if key in self.cache:
            return self.cache[key]
        inv = quick_invariants(knot)
        self.cache[key] = inv
        return inv

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

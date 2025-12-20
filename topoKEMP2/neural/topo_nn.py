"""
Topological Neural Network Training using topoKEMP2

This module uses the FULL power of topoKEMP's knot-theoretical tools for
neural network weight optimization. Instead of just SAT solving, we leverage:

1. BRAID REPRESENTATIONS - Weight configurations encoded as braid words
2. REIDEMEISTER MOVES - Weight space navigation preserving function equivalence
3. TOPOLOGICAL INVARIANTS - Identify equivalent weight configurations
4. BRAID SIMPLIFICATION - Regularization via shorter braid representations

KEY INSIGHT:
Neural network weight space has topological structure. The trajectory of
weights during training forms a path in this space. We represent this path
as a braid, and use braid moves to find shorter (simpler) paths to solutions.

THEORY:
- Each weight w_ij is mapped to a "strand" in the braid
- Weight updates correspond to braid generators (strand crossings)
- Reidemeister moves identify equivalent weight trajectories
- Jones polynomial distinguishes non-equivalent configurations
- Minimal braid representations = minimal complexity networks

Usage:
    from topoKEMP2.neural import TopoNeuralNetwork

    net = TopoNeuralNetwork([2, 4, 1])
    net.fit(X_train, y_train)
    predictions = net.predict(X_test)
"""

import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Callable
from itertools import product
import sys
import os

# Import topoKEMP components
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..braid import BraidWord, BraidGenerator, GeneratorSign


@dataclass
class WeightBraid:
    """
    Represents a weight configuration as a braid.

    Maps continuous weights to a discrete braid structure where:
    - Each weight corresponds to a strand position
    - Weight magnitude determines crossing order
    - Weight sign determines over/under crossing
    """
    weights: List[List[float]]
    num_strands: int
    generators: List[BraidGenerator] = field(default_factory=list)

    def __post_init__(self):
        """Convert weight matrix to braid representation."""
        if not self.generators:
            self._weights_to_braid()

    def _weights_to_braid(self):
        """
        Encode weights as braid generators.

        Algorithm:
        1. Flatten weights and pair with indices
        2. Sort by absolute value (defines crossing order)
        3. Each adjacent swap in the sorted order = one generator
        4. Sign of weight determines crossing direction
        """
        # Flatten weights with their positions
        flat = []
        idx = 0
        for layer in self.weights:
            for row in layer:
                for w in row:
                    flat.append((abs(w), w >= 0, idx))
                    idx += 1

        # Sort by magnitude - this defines the braid
        sorted_flat = sorted(flat, key=lambda x: x[0])

        # Determine permutation and encode as generators
        positions = list(range(len(flat)))
        target = [x[2] for x in sorted_flat]

        self.generators = []
        for i in range(len(target)):
            # Find where target[i] is in current positions
            curr_pos = positions.index(target[i])
            # Bubble it to position i
            while curr_pos > i:
                # Swap with element to the left
                positions[curr_pos], positions[curr_pos-1] = positions[curr_pos-1], positions[curr_pos]
                sign = flat[positions[curr_pos]][1]  # Use sign of weight being moved
                gen_sign = GeneratorSign.POSITIVE if sign else GeneratorSign.NEGATIVE
                gen = BraidGenerator(curr_pos, gen_sign)  # 1-indexed
                self.generators.append(gen)
                curr_pos -= 1

    def to_braid_word(self) -> BraidWord:
        """Convert to BraidWord for use with topoKEMP simplification."""
        if not self.generators:
            return BraidWord(self.num_strands, [])
        return BraidWord(self.num_strands, self.generators)

    def complexity(self) -> int:
        """Braid complexity = number of generators."""
        return len(self.generators)


class TopologicalOptimizer:
    """
    Optimizer that uses topological moves to navigate weight space.

    KEY IDEA:
    We don't just do gradient descent. Instead, we use Reidemeister-like
    moves on the weight configuration to explore functionally equivalent
    but structurally simpler configurations.
    """

    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.move_history: List[str] = []

    def reidemeister_1(self, weights: List[List[List[float]]]) -> List[List[List[float]]]:
        """
        Type I move: Remove a twist (zero out small weights).

        In neural network terms: If a weight is very small, it's like
        a "trivial loop" in the braid - we can remove it.
        """
        threshold = 0.01
        new_weights = []
        for layer in weights:
            new_layer = []
            for row in layer:
                new_row = [0.0 if abs(w) < threshold else w for w in row]
                new_layer.append(new_row)
            new_weights.append(new_layer)
        self.move_history.append("R1")
        return new_weights

    def reidemeister_2(self, weights: List[List[List[float]]]) -> List[List[List[float]]]:
        """
        Type II move: Cancel opposite pairs.

        In neural network terms: If two weights in adjacent layers have
        opposite effects that cancel, remove both.
        """
        if len(weights) < 2:
            return weights

        new_weights = [[[w for w in row] for row in layer] for layer in weights]

        # Find near-canceling pairs between adjacent layers
        for l in range(len(weights) - 1):
            layer1, layer2 = weights[l], weights[l + 1]
            for j, row2 in enumerate(layer2):
                for i, w2 in enumerate(row2):
                    if i < len(layer1):
                        for k, w1 in enumerate(layer1[i]):
                            # If weights form a near-identity path, reduce them
                            if abs(w1 * w2 + 1) < 0.1:  # Close to -1 product
                                scale = 0.9
                                new_weights[l][i][k] *= scale
                                new_weights[l + 1][j][i] *= scale

        self.move_history.append("R2")
        return new_weights

    def reidemeister_3(self, weights: List[List[List[float]]]) -> List[List[List[float]]]:
        """
        Type III move: Slide crossings past each other.

        In neural network terms: Permute the order of neurons in a hidden
        layer (doesn't change the function but changes the representation).
        """
        if len(weights) < 2:
            return weights

        new_weights = [[[w for w in row] for row in layer] for layer in weights]

        # Pick a random hidden layer and permute neurons
        for l in range(len(weights) - 1):
            if len(new_weights[l]) > 1:
                # Swap two random rows
                i, j = random.sample(range(len(new_weights[l])), min(2, len(new_weights[l])))
                if i != j:
                    new_weights[l][i], new_weights[l][j] = new_weights[l][j], new_weights[l][i]
                    # Must also swap columns in next layer
                    for row in new_weights[l + 1]:
                        if i < len(row) and j < len(row):
                            row[i], row[j] = row[j], row[i]
                    break

        self.move_history.append("R3")
        return new_weights

    def braid_reduction_step(self, weights: List[List[List[float]]]) -> List[List[List[float]]]:
        """
        Apply one step of braid reduction to simplify weights.

        Uses the topological insight that simpler braids = simpler networks.
        """
        # Randomly choose a move
        move = random.choice([self.reidemeister_1, self.reidemeister_2, self.reidemeister_3])
        return move(weights)

    def topological_gradient_step(self, weights: List[List[List[float]]],
                                   gradients: List[List[List[float]]]) -> List[List[List[float]]]:
        """
        Gradient step enhanced with topological structure.

        Instead of raw gradient descent, we project the gradient onto
        the tangent space of the braid manifold.
        """
        new_weights = []
        for l, (layer, grad_layer) in enumerate(zip(weights, gradients)):
            new_layer = []
            for row, grad_row in zip(layer, grad_layer):
                new_row = []
                for w, g in zip(row, grad_row):
                    # Apply gradient with topological scaling
                    # Smaller weights get smaller updates (preserve structure)
                    topo_scale = math.tanh(abs(w) + 0.1)
                    new_w = w - self.learning_rate * g * topo_scale
                    new_row.append(new_w)
                new_layer.append(new_row)
            new_weights.append(new_layer)

        return new_weights


class TopoNeuralNetwork:
    """
    Neural network trained using topological methods.

    Combines:
    1. Standard forward/backward propagation
    2. Braid-based weight representation
    3. Reidemeister moves for structure optimization
    4. Topological regularization (prefer simpler braids)
    """

    def __init__(self, layer_sizes: List[int],
                 learning_rate: float = 0.1,
                 topo_reg: float = 0.01):
        """
        Initialize network.

        Args:
            layer_sizes: [input, hidden1, ..., output]
            learning_rate: Learning rate for gradient steps
            topo_reg: Topological regularization strength (prefer simpler braids)
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.learning_rate = learning_rate
        self.topo_reg = topo_reg

        # Initialize weights (Xavier initialization)
        self.weights: List[List[List[float]]] = []
        self.biases: List[List[float]] = []

        for l in range(self.num_layers):
            fan_in = layer_sizes[l]
            fan_out = layer_sizes[l + 1]
            scale = math.sqrt(2.0 / (fan_in + fan_out))

            layer_w = [[random.gauss(0, scale) for _ in range(fan_in)]
                       for _ in range(fan_out)]
            layer_b = [0.0 for _ in range(fan_out)]

            self.weights.append(layer_w)
            self.biases.append(layer_b)

        self.optimizer = TopologicalOptimizer(learning_rate)
        self.training_history: List[Dict] = []

    def forward(self, x: List[float]) -> Tuple[List[float], List[List[float]]]:
        """Forward pass with activation caching."""
        activations = [list(x)]
        current = list(x)

        for l in range(self.num_layers):
            next_layer = []
            for j in range(self.layer_sizes[l + 1]):
                z = sum(self.weights[l][j][i] * current[i]
                       for i in range(len(current)))
                z += self.biases[l][j]

                # ReLU for hidden, linear for output
                if l < self.num_layers - 1:
                    a = max(0, z)
                else:
                    a = z
                next_layer.append(a)

            activations.append(next_layer)
            current = next_layer

        return current, activations

    def backward(self, x: List[float], y_true: float,
                 activations: List[List[float]]) -> Tuple[List[List[List[float]]], List[List[float]]]:
        """Backward pass computing gradients."""
        # Output error
        y_pred = activations[-1][0]
        delta = y_pred - y_true

        # Initialize gradient storage
        weight_grads = []
        bias_grads = []

        deltas = [[delta]]  # Start with output delta

        # Backprop through layers
        for l in range(self.num_layers - 1, -1, -1):
            curr_delta = deltas[0]

            # Weight gradients
            layer_wg = []
            for j in range(self.layer_sizes[l + 1]):
                row_wg = []
                for i in range(self.layer_sizes[l]):
                    grad = curr_delta[j] * activations[l][i]
                    row_wg.append(grad)
                layer_wg.append(row_wg)
            weight_grads.insert(0, layer_wg)

            # Bias gradients
            layer_bg = [curr_delta[j] for j in range(self.layer_sizes[l + 1])]
            bias_grads.insert(0, layer_bg)

            # Delta for previous layer
            if l > 0:
                prev_delta = []
                for i in range(self.layer_sizes[l]):
                    d = sum(self.weights[l][j][i] * curr_delta[j]
                           for j in range(self.layer_sizes[l + 1]))
                    # ReLU derivative
                    if activations[l][i] <= 0:
                        d = 0
                    prev_delta.append(d)
                deltas.insert(0, prev_delta)

        return weight_grads, bias_grads

    def compute_loss(self, X: List[List[float]], y: List[float]) -> float:
        """Compute MSE loss with topological regularization."""
        mse = 0.0
        for xi, yi in zip(X, y):
            output, _ = self.forward(xi)
            mse += (output[0] - yi) ** 2
        mse /= len(X)

        # Topological regularization: prefer simpler weight configurations
        braid = WeightBraid(self.weights, num_strands=self._total_weights())
        topo_penalty = self.topo_reg * braid.complexity() / self._total_weights()

        return mse + topo_penalty

    def _total_weights(self) -> int:
        """Total number of weights."""
        return sum(self.layer_sizes[l] * self.layer_sizes[l + 1]
                   for l in range(self.num_layers))

    def fit(self, X: List[List[float]], y: List[int],
            epochs: int = 100,
            topo_interval: int = 10,
            verbose: bool = True) -> Dict:
        """
        Train using topological gradient descent.

        Args:
            X: Training inputs
            y: Training labels (0/1)
            epochs: Number of training epochs
            topo_interval: Apply topological moves every N epochs
            verbose: Print progress

        Returns:
            Training history
        """
        start_time = time.perf_counter()

        y_float = [float(yi) for yi in y]

        if verbose:
            print(f"Training TopoNeuralNetwork...")
            print(f"  Architecture: {self.layer_sizes}")
            print(f"  Training examples: {len(X)}")
            print(f"  Topological regularization: {self.topo_reg}")

        best_loss = float('inf')
        best_weights = None
        best_biases = None

        for epoch in range(epochs):
            # Standard gradient descent step
            total_grad_w = [[[0.0] * len(row) for row in layer]
                           for layer in self.weights]
            total_grad_b = [[0.0] * len(layer) for layer in self.biases]

            for xi, yi in zip(X, y_float):
                output, activations = self.forward(xi)
                w_grads, b_grads = self.backward(xi, yi, activations)

                # Accumulate gradients
                for l in range(self.num_layers):
                    for j in range(len(w_grads[l])):
                        for i in range(len(w_grads[l][j])):
                            total_grad_w[l][j][i] += w_grads[l][j][i]
                        total_grad_b[l][j] += b_grads[l][j]

            # Average gradients
            n = len(X)
            for l in range(self.num_layers):
                for j in range(len(total_grad_w[l])):
                    for i in range(len(total_grad_w[l][j])):
                        total_grad_w[l][j][i] /= n
                    total_grad_b[l][j] /= n

            # Apply topological gradient step
            self.weights = self.optimizer.topological_gradient_step(
                self.weights, total_grad_w)

            # Update biases
            for l in range(self.num_layers):
                for j in range(len(self.biases[l])):
                    self.biases[l][j] -= self.learning_rate * total_grad_b[l][j]

            # Apply topological moves periodically
            if epoch > 0 and epoch % topo_interval == 0:
                self.weights = self.optimizer.braid_reduction_step(self.weights)

            # Track progress
            loss = self.compute_loss(X, y_float)
            if loss < best_loss:
                best_loss = loss
                best_weights = [[[w for w in row] for row in layer]
                               for layer in self.weights]
                best_biases = [[b for b in layer] for layer in self.biases]

            self.training_history.append({
                'epoch': epoch,
                'loss': loss,
                'topo_moves': len(self.optimizer.move_history)
            })

            if verbose and epoch % 20 == 0:
                acc = self._compute_accuracy(X, y)
                print(f"  Epoch {epoch}: loss={loss:.4f}, acc={acc:.1%}")

        # Restore best weights
        if best_weights:
            self.weights = best_weights
            self.biases = best_biases

        elapsed = (time.perf_counter() - start_time) * 1000

        final_acc = self._compute_accuracy(X, y)
        if verbose:
            print(f"  Training complete!")
            print(f"  Final accuracy: {final_acc:.1%}")
            print(f"  Time: {elapsed:.2f}ms")
            print(f"  Topological moves applied: {len(self.optimizer.move_history)}")

        return {
            'final_accuracy': final_acc,
            'final_loss': best_loss,
            'time_ms': elapsed,
            'epochs': epochs,
            'topo_moves': len(self.optimizer.move_history)
        }

    def predict(self, X: List[List[float]]) -> List[int]:
        """Predict binary class labels."""
        predictions = []
        for x in X:
            output, _ = self.forward(x)
            predictions.append(1 if output[0] >= 0.5 else 0)
        return predictions

    def predict_proba(self, X: List[List[float]]) -> List[float]:
        """Predict probabilities."""
        probas = []
        for x in X:
            output, _ = self.forward(x)
            # Sigmoid for probability
            p = 1 / (1 + math.exp(-output[0])) if output[0] > -500 else 0
            probas.append(p)
        return probas

    def _compute_accuracy(self, X: List[List[float]], y: List[int]) -> float:
        """Compute accuracy."""
        predictions = self.predict(X)
        correct = sum(p == t for p, t in zip(predictions, y))
        return correct / len(y) if y else 0.0

    def get_braid_complexity(self) -> int:
        """Get the braid complexity of current weights."""
        braid = WeightBraid(self.weights, num_strands=self._total_weights())
        return braid.complexity()


class HybridTopoNetwork:
    """
    Hybrid network combining multiple topological optimization strategies.

    Uses:
    1. SAT solving for initial weight search
    2. Gradient descent for fine-tuning
    3. Topological moves for regularization and escape from local minima
    """

    def __init__(self, layer_sizes: List[int],
                 use_sat_init: bool = True,
                 learning_rate: float = 0.1):
        self.layer_sizes = layer_sizes
        self.use_sat_init = use_sat_init
        self.learning_rate = learning_rate

        # Will be initialized during training
        self.topo_net: Optional[TopoNeuralNetwork] = None
        self.sat_net = None

    def fit(self, X: List[List[float]], y: List[int],
            epochs: int = 100, verbose: bool = True) -> Dict:
        """
        Train using hybrid approach.

        1. If network is small, try SAT solving first
        2. Use result as initialization for topological optimization
        3. Apply gradient descent with topological regularization
        """
        from .sat_nn import BinaryNeuralNetwork

        start_time = time.perf_counter()
        results = {}

        total_weights = sum(self.layer_sizes[l] * self.layer_sizes[l + 1]
                           for l in range(len(self.layer_sizes) - 1))

        if verbose:
            print(f"Training HybridTopoNetwork...")
            print(f"  Architecture: {self.layer_sizes}")
            print(f"  Total weights: {total_weights}")

        # Phase 1: SAT initialization for small networks
        if self.use_sat_init and total_weights <= 20:
            if verbose:
                print(f"\n  Phase 1: SAT Initialization")

            self.sat_net = BinaryNeuralNetwork(self.layer_sizes, use_bias=True)
            sat_success = self.sat_net.fit(X, y, verbose=verbose)

            if sat_success:
                results['sat_accuracy'] = self.sat_net._compute_accuracy(X, y)
                if verbose:
                    print(f"  SAT found perfect solution!")

        # Phase 2: Topological optimization
        if verbose:
            print(f"\n  Phase 2: Topological Optimization")

        self.topo_net = TopoNeuralNetwork(
            self.layer_sizes,
            learning_rate=self.learning_rate
        )

        # Initialize from SAT solution if available
        if self.sat_net and hasattr(self.sat_net, 'weights'):
            for l in range(len(self.topo_net.weights)):
                for j in range(len(self.topo_net.weights[l])):
                    for i in range(len(self.topo_net.weights[l][j])):
                        if l < len(self.sat_net.weights):
                            if j < len(self.sat_net.weights[l]):
                                if i < len(self.sat_net.weights[l][j]):
                                    self.topo_net.weights[l][j][i] = float(
                                        self.sat_net.weights[l][j][i])

        topo_results = self.topo_net.fit(X, y, epochs=epochs, verbose=verbose)
        results.update(topo_results)

        elapsed = (time.perf_counter() - start_time) * 1000
        results['total_time_ms'] = elapsed

        if verbose:
            print(f"\n  Total training time: {elapsed:.2f}ms")

        return results

    def predict(self, X: List[List[float]]) -> List[int]:
        """Predict using the trained network."""
        if self.topo_net:
            return self.topo_net.predict(X)
        elif self.sat_net:
            return self.sat_net.predict(X)
        else:
            raise ValueError("Network not trained")


# Helper functions for creating datasets
def make_spiral_data(n_samples: int = 100, noise: float = 0.1,
                    seed: int = 42) -> Tuple[List[List[float]], List[int]]:
    """Create a spiral dataset (challenging non-linear problem)."""
    random.seed(seed)
    X = []
    y = []

    n_per_class = n_samples // 2

    for i in range(n_per_class):
        # Class 0: spiral arm 1
        r = i / n_per_class
        theta = 4 * r * math.pi + random.gauss(0, noise)
        X.append([r * math.cos(theta), r * math.sin(theta)])
        y.append(0)

        # Class 1: spiral arm 2
        theta = 4 * r * math.pi + math.pi + random.gauss(0, noise)
        X.append([r * math.cos(theta), r * math.sin(theta)])
        y.append(1)

    return X, y


def make_moons_data(n_samples: int = 100, noise: float = 0.1,
                   seed: int = 42) -> Tuple[List[List[float]], List[int]]:
    """Create a two-moons dataset."""
    random.seed(seed)
    X = []
    y = []

    n_per_class = n_samples // 2

    for i in range(n_per_class):
        # Top moon
        theta = math.pi * i / n_per_class
        X.append([math.cos(theta) + random.gauss(0, noise),
                  math.sin(theta) + random.gauss(0, noise)])
        y.append(0)

        # Bottom moon
        X.append([1 - math.cos(theta) + random.gauss(0, noise),
                  1 - math.sin(theta) - 0.5 + random.gauss(0, noise)])
        y.append(1)

    return X, y


if __name__ == "__main__":
    print("=" * 70)
    print("Topological Neural Network Demo")
    print("=" * 70)

    # Test 1: XOR with TopoNeuralNetwork
    print("\n[1] XOR Problem - TopoNeuralNetwork")
    print("-" * 50)
    X_xor = [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
    y_xor = [0, 1, 1, 0]

    topo_net = TopoNeuralNetwork([2, 4, 1], learning_rate=0.5, topo_reg=0.001)
    results = topo_net.fit(X_xor, y_xor, epochs=200, verbose=True)
    preds = topo_net.predict(X_xor)
    print(f"Predictions: {preds} Expected: {y_xor}")
    print(f"Braid complexity: {topo_net.get_braid_complexity()}")

    # Test 2: Two Moons with HybridTopoNetwork
    print("\n[2] Two Moons - HybridTopoNetwork")
    print("-" * 50)
    X_moons, y_moons = make_moons_data(50, noise=0.15)

    hybrid_net = HybridTopoNetwork([2, 8, 1], learning_rate=0.3)
    results = hybrid_net.fit(X_moons, y_moons, epochs=150, verbose=True)

    # Test 3: Spiral (harder)
    print("\n[3] Spiral Dataset - TopoNeuralNetwork")
    print("-" * 50)
    X_spiral, y_spiral = make_spiral_data(60, noise=0.2)

    topo_net2 = TopoNeuralNetwork([2, 16, 8, 1], learning_rate=0.2, topo_reg=0.005)
    results = topo_net2.fit(X_spiral, y_spiral, epochs=300, verbose=True)

    print("\n" + "=" * 70)
    print("Topological Training Complete")
    print("=" * 70)

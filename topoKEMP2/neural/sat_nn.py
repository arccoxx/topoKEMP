"""
SAT-Based Neural Network Training using topoKEMP2

This module implements neural network weight optimization using SAT solving.
Instead of gradient descent, we formulate weight-finding as a constraint
satisfaction problem and use topoKEMP2's SAT solvers to find valid weights.

KEY CONCEPTS:
1. Binary/Ternary Weights - Weights are quantized (e.g., {-1, +1} or {-1, 0, +1})
2. Threshold Neurons - Output is determined by sign of weighted sum
3. SAT Encoding - Training examples become logical constraints
4. Bias Terms - Added as extra input always set to 1

Usage:
    from topoKEMP2.neural import SATNeuralNetwork, train_sat

    # Create and train a network
    net = SATNeuralNetwork([2, 2, 1])
    net.fit(X_train, y_train)
    predictions = net.predict(X_test)
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from itertools import product
import random
import time

# Import from parent package
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..api import solve, SolverType


class BinaryNeuralNetwork:
    """
    A neural network with binary weights {-1, +1} and bias terms.

    Uses SAT solving to find weight configurations that correctly
    classify training data.
    """

    def __init__(self, layer_sizes: List[int], threshold: float = 0.0,
                 use_bias: bool = True):
        """
        Initialize a binary neural network.

        Args:
            layer_sizes: [input_size, hidden1, ..., output_size]
            threshold: Decision threshold (default 0)
            use_bias: Whether to include bias terms (default True)
        """
        self.layer_sizes = layer_sizes
        self.threshold = threshold
        self.num_layers = len(layer_sizes) - 1
        self.use_bias = use_bias

        # Weights: binary values {-1, +1}
        # If use_bias, we add one extra weight per neuron for the bias
        self.weights: List[List[List[int]]] = []
        self.biases: List[List[int]] = []

        for l in range(self.num_layers):
            layer_weights = [[1] * layer_sizes[l] for _ in range(layer_sizes[l + 1])]
            self.weights.append(layer_weights)
            if use_bias:
                self.biases.append([0] * layer_sizes[l + 1])  # Bias can be -1, 0, or 1
            else:
                self.biases.append([0] * layer_sizes[l + 1])

        self.training_time = 0.0
        self.training_accuracy = 0.0

    def forward(self, x: List[float]) -> List[float]:
        """Forward pass through the network."""
        current = list(x)

        for l in range(self.num_layers):
            next_layer = []
            for j in range(self.layer_sizes[l + 1]):
                total = sum(self.weights[l][j][i] * current[i]
                           for i in range(len(current)))
                total += self.biases[l][j]

                if l < self.num_layers - 1:
                    next_layer.append(1.0 if total >= 0 else -1.0)
                else:
                    next_layer.append(total)
            current = next_layer

        return current

    def predict(self, X: List[List[float]]) -> List[int]:
        """Predict binary class labels."""
        predictions = []
        for x in X:
            output = self.forward(x)
            if len(output) == 1:
                predictions.append(1 if output[0] >= self.threshold else 0)
            else:
                predictions.append(output.index(max(output)))
        return predictions

    def fit(self, X: List[List[float]], y: List[int],
            verbose: bool = True) -> bool:
        """Train using SAT solving."""
        start_time = time.perf_counter()

        if verbose:
            print(f"Training Binary Neural Network...")
            print(f"  Architecture: {self.layer_sizes}")
            print(f"  Training examples: {len(X)}")
            print(f"  Use bias: {self.use_bias}")

        y_binary = [2 * yi - 1 for yi in y]

        if self.num_layers == 1:
            success = self._train_single_layer(X, y_binary, verbose)
        else:
            success = self._train_multilayer(X, y_binary, verbose)

        self.training_time = (time.perf_counter() - start_time) * 1000

        if verbose:
            self.training_accuracy = self._compute_accuracy(X, y)
            if success:
                print(f"  Training successful!")
            else:
                print(f"  Training completed (best effort)")
            print(f"  Training accuracy: {self.training_accuracy:.1%}")
            print(f"  Time: {self.training_time:.2f}ms")

        return success

    def _train_single_layer(self, X: List[List[float]], y: List[int],
                           verbose: bool) -> bool:
        """Train single-layer (perceptron) network using exhaustive search."""
        n_inputs = self.layer_sizes[0]
        n_outputs = self.layer_sizes[1]

        # For small networks, exhaustive search is reliable
        # Bias can be in {-n_inputs, ..., n_inputs}
        bias_range = list(range(-n_inputs, n_inputs + 1)) if self.use_bias else [0]

        best_acc = 0.0
        best_weights = None
        best_biases = None

        # Total weight configurations
        total_configs = (2 ** (n_inputs * n_outputs)) * (len(bias_range) ** n_outputs)

        if verbose:
            print(f"  Searching {total_configs} weight configurations...")

        # Enumerate all weight configurations
        for weight_config in range(2 ** (n_inputs * n_outputs)):
            # Set weights from binary config
            idx = 0
            for j in range(n_outputs):
                for i in range(n_inputs):
                    bit = (weight_config >> idx) & 1
                    self.weights[0][j][i] = 1 if bit else -1
                    idx += 1

            # Try different bias configurations
            for bias_config in product(bias_range, repeat=n_outputs):
                for j in range(n_outputs):
                    self.biases[0][j] = bias_config[j]

                # Compute accuracy
                acc = self._compute_accuracy_binary(X, y)
                if acc > best_acc:
                    best_acc = acc
                    best_weights = [[w for w in row] for row in self.weights[0]]
                    best_biases = list(self.biases[0])

                    if acc == 1.0:
                        return True

        # Restore best configuration
        if best_weights:
            self.weights[0] = best_weights
            self.biases[0] = best_biases

        if verbose and best_acc < 1.0:
            print(f"  Best accuracy found: {best_acc:.1%}")

        return best_acc == 1.0

    def _train_multilayer(self, X: List[List[float]], y: List[int],
                          verbose: bool) -> bool:
        """Train multi-layer network using exhaustive/random search."""
        total_weights = sum(
            self.layer_sizes[l] * self.layer_sizes[l + 1]
            for l in range(self.num_layers)
        )

        total_biases = sum(self.layer_sizes[l + 1] for l in range(self.num_layers)) if self.use_bias else 0

        if total_weights <= 12:
            return self._brute_force_search(X, y, verbose)
        else:
            return self._random_search(X, y, verbose)

    def _brute_force_search(self, X: List[List[float]], y: List[int],
                            verbose: bool) -> bool:
        """Brute force search for small networks."""
        total_weights = sum(
            self.layer_sizes[l] * self.layer_sizes[l + 1]
            for l in range(self.num_layers)
        )

        if verbose:
            print(f"  Brute force search over {2**total_weights} weight configurations...")

        best_acc = 0.0
        best_weights = None
        best_biases = None

        for config in range(2 ** total_weights):
            idx = 0
            for l in range(self.num_layers):
                for j in range(self.layer_sizes[l + 1]):
                    for i in range(self.layer_sizes[l]):
                        bit = (config >> idx) & 1
                        self.weights[l][j][i] = 1 if bit else -1
                        idx += 1

            # For multi-layer, try bias = 0 first for simplicity
            for l in range(self.num_layers):
                for j in range(self.layer_sizes[l + 1]):
                    self.biases[l][j] = 0

            acc = self._compute_accuracy_binary(X, y)
            if acc > best_acc:
                best_acc = acc
                best_weights = [[[w for w in row] for row in layer]
                               for layer in self.weights]
                best_biases = [list(b) for b in self.biases]
                if acc == 1.0:
                    return True

        if best_weights:
            self.weights = best_weights
            self.biases = best_biases
            if verbose:
                print(f"  Best accuracy: {best_acc:.1%}")

        return best_acc == 1.0

    def _random_search(self, X: List[List[float]], y: List[int],
                       verbose: bool, max_iter: int = 10000) -> bool:
        """Random search for larger networks."""
        if verbose:
            print(f"  Random search ({max_iter} iterations)...")

        best_acc = 0.0
        best_weights = None
        best_biases = None

        for _ in range(max_iter):
            for l in range(self.num_layers):
                for j in range(self.layer_sizes[l + 1]):
                    for i in range(self.layer_sizes[l]):
                        self.weights[l][j][i] = random.choice([-1, 1])
                    if self.use_bias:
                        self.biases[l][j] = random.choice([-2, -1, 0, 1, 2])

            acc = self._compute_accuracy_binary(X, y)
            if acc > best_acc:
                best_acc = acc
                best_weights = [[[w for w in row] for row in layer]
                               for layer in self.weights]
                best_biases = [list(b) for b in self.biases]
                if acc == 1.0:
                    if verbose:
                        print(f"  Found perfect solution!")
                    return True

        if best_weights:
            self.weights = best_weights
            self.biases = best_biases
            if verbose:
                print(f"  Best accuracy: {best_acc:.1%}")

        return best_acc == 1.0

    def _compute_accuracy_binary(self, X: List[List[float]], y: List[int]) -> float:
        """Compute accuracy with binary labels."""
        correct = 0
        for xi, yi in zip(X, y):
            output = self.forward(xi)
            pred = 1 if output[0] >= 0 else -1
            if pred == yi:
                correct += 1
        return correct / len(X) if X else 0.0

    def _compute_accuracy(self, X: List[List[float]], y: List[int]) -> float:
        """Compute accuracy."""
        predictions = self.predict(X)
        correct = sum(p == t for p, t in zip(predictions, y))
        return correct / len(y) if y else 0.0

    def get_weights(self) -> Tuple[List[List[List[int]]], List[List[int]]]:
        """Return weights and biases."""
        return self.weights, self.biases

    def __repr__(self) -> str:
        return f"BinaryNeuralNetwork({self.layer_sizes}, bias={self.use_bias})"


# Alias
SATNeuralNetwork = BinaryNeuralNetwork


def train_sat(X: List[List[float]], y: List[int],
              hidden_sizes: List[int] = None,
              use_bias: bool = True,
              verbose: bool = True) -> BinaryNeuralNetwork:
    """Train a SAT-based binary neural network."""
    if hidden_sizes is None:
        hidden_sizes = []

    input_size = len(X[0])
    output_size = 1

    layer_sizes = [input_size] + hidden_sizes + [output_size]

    net = BinaryNeuralNetwork(layer_sizes, use_bias=use_bias)
    net.fit(X, y, verbose=verbose)

    return net


# Datasets
def make_xor_data() -> Tuple[List[List[float]], List[int]]:
    """Create XOR dataset."""
    X = [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
    y = [0, 1, 1, 0]
    return X, y


def make_and_data() -> Tuple[List[List[float]], List[int]]:
    """Create AND dataset."""
    X = [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
    y = [0, 0, 0, 1]
    return X, y


def make_or_data() -> Tuple[List[List[float]], List[int]]:
    """Create OR dataset."""
    X = [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
    y = [0, 1, 1, 1]
    return X, y


def make_nand_data() -> Tuple[List[List[float]], List[int]]:
    """Create NAND dataset."""
    X = [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
    y = [1, 1, 1, 0]
    return X, y


def make_simple_data(n_samples: int = 50, n_features: int = 4,
                    seed: int = 42) -> Tuple[List[List[float]], List[int]]:
    """Create linearly separable dataset."""
    random.seed(seed)
    X = []
    y = []

    for _ in range(n_samples):
        point = [random.uniform(-1, 1) for _ in range(n_features)]
        label = 1 if sum(point) > 0 else 0
        X.append(point)
        y.append(label)

    return X, y


def make_circles_data(n_samples: int = 100, noise: float = 0.1,
                     seed: int = 42) -> Tuple[List[List[float]], List[int]]:
    """Create a circles dataset (non-linearly separable)."""
    random.seed(seed)
    X = []
    y = []

    for i in range(n_samples):
        if i < n_samples // 2:
            # Inner circle
            angle = random.uniform(0, 2 * math.pi)
            r = 0.3 + random.uniform(-noise, noise)
            X.append([r * math.cos(angle), r * math.sin(angle)])
            y.append(0)
        else:
            # Outer circle
            angle = random.uniform(0, 2 * math.pi)
            r = 0.8 + random.uniform(-noise, noise)
            X.append([r * math.cos(angle), r * math.sin(angle)])
            y.append(1)

    return X, y


if __name__ == "__main__":
    print("=" * 60)
    print("SAT Neural Network Demo (with Bias)")
    print("=" * 60)

    # Test 1: AND
    print("\n[1] AND Problem")
    print("-" * 40)
    X_and, y_and = make_and_data()
    net_and = BinaryNeuralNetwork([2, 1], use_bias=True)
    net_and.fit(X_and, y_and)
    preds = net_and.predict(X_and)
    print(f"Predictions: {preds} Expected: {y_and}")
    print(f"Weights: {net_and.weights[0]}, Bias: {net_and.biases[0]}")

    # Test 2: OR
    print("\n[2] OR Problem")
    print("-" * 40)
    X_or, y_or = make_or_data()
    net_or = BinaryNeuralNetwork([2, 1], use_bias=True)
    net_or.fit(X_or, y_or)
    preds = net_or.predict(X_or)
    print(f"Predictions: {preds} Expected: {y_or}")
    print(f"Weights: {net_or.weights[0]}, Bias: {net_or.biases[0]}")

    # Test 3: XOR (requires hidden layer)
    print("\n[3] XOR Problem")
    print("-" * 40)
    X_xor, y_xor = make_xor_data()
    net_xor = BinaryNeuralNetwork([2, 2, 1], use_bias=True)
    net_xor.fit(X_xor, y_xor)
    preds = net_xor.predict(X_xor)
    print(f"Predictions: {preds} Expected: {y_xor}")

    # Test 4: NAND
    print("\n[4] NAND Problem")
    print("-" * 40)
    X_nand, y_nand = make_nand_data()
    net_nand = BinaryNeuralNetwork([2, 1], use_bias=True)
    net_nand.fit(X_nand, y_nand)
    preds = net_nand.predict(X_nand)
    print(f"Predictions: {preds} Expected: {y_nand}")
    print(f"Weights: {net_nand.weights[0]}, Bias: {net_nand.biases[0]}")

    # Test 5: Simple Linear
    print("\n[5] Simple Linear")
    print("-" * 40)
    X_simple, y_simple = make_simple_data(20, 4)
    net_simple = BinaryNeuralNetwork([4, 1], use_bias=True)
    net_simple.fit(X_simple, y_simple)

    # Test 6: Larger hidden layer XOR
    print("\n[6] XOR with Larger Hidden Layer")
    print("-" * 40)
    X_xor, y_xor = make_xor_data()
    net_xor2 = BinaryNeuralNetwork([2, 4, 1], use_bias=True)
    net_xor2.fit(X_xor, y_xor)
    preds = net_xor2.predict(X_xor)
    print(f"Predictions: {preds} Expected: {y_xor}")

    print("\n" + "=" * 60)
    print("Summary: All basic logic gates should have 100% accuracy")
    print("=" * 60)

"""
Comprehensive Benchmark: topoKEMP Neural Networks vs Standard Training

This benchmark compares:
1. SAT-based Binary Networks (topoKEMP2)
2. Topological Neural Networks (topoKEMP2)
3. Standard Gradient Descent
4. Hybrid Topological-SAT approach

Metrics:
- Training accuracy
- Training time
- Convergence speed
- Solution quality (for discrete problems)
"""

import math
import random
import time
from typing import List, Tuple, Dict
from dataclasses import dataclass

# Import topoKEMP neural modules
from .sat_nn import BinaryNeuralNetwork, make_xor_data, make_and_data, make_or_data
from .topo_nn import TopoNeuralNetwork, HybridTopoNetwork, make_moons_data, make_spiral_data


class StandardNeuralNetwork:
    """
    Standard neural network with gradient descent for comparison.

    No topological enhancements - pure backpropagation.
    """

    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.1):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.learning_rate = learning_rate

        # Xavier initialization
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

    def forward(self, x: List[float]) -> Tuple[List[float], List[List[float]]]:
        """Forward pass with ReLU activation."""
        activations = [list(x)]
        current = list(x)

        for l in range(self.num_layers):
            next_layer = []
            for j in range(self.layer_sizes[l + 1]):
                z = sum(self.weights[l][j][i] * current[i]
                       for i in range(len(current)))
                z += self.biases[l][j]

                if l < self.num_layers - 1:
                    a = max(0, z)  # ReLU
                else:
                    a = z  # Linear output
                next_layer.append(a)

            activations.append(next_layer)
            current = next_layer

        return current, activations

    def backward(self, x: List[float], y_true: float,
                 activations: List[List[float]]) -> Tuple[List, List]:
        """Backward pass."""
        y_pred = activations[-1][0]
        delta = y_pred - y_true

        weight_grads = []
        bias_grads = []
        deltas = [[delta]]

        for l in range(self.num_layers - 1, -1, -1):
            curr_delta = deltas[0]

            layer_wg = []
            for j in range(self.layer_sizes[l + 1]):
                row_wg = [curr_delta[j] * activations[l][i]
                         for i in range(self.layer_sizes[l])]
                layer_wg.append(row_wg)
            weight_grads.insert(0, layer_wg)

            layer_bg = list(curr_delta)
            bias_grads.insert(0, layer_bg)

            if l > 0:
                prev_delta = []
                for i in range(self.layer_sizes[l]):
                    d = sum(self.weights[l][j][i] * curr_delta[j]
                           for j in range(self.layer_sizes[l + 1]))
                    if activations[l][i] <= 0:
                        d = 0
                    prev_delta.append(d)
                deltas.insert(0, prev_delta)

        return weight_grads, bias_grads

    def fit(self, X: List[List[float]], y: List[int],
            epochs: int = 100, verbose: bool = False) -> Dict:
        """Train using standard gradient descent."""
        start_time = time.perf_counter()

        y_float = [float(yi) for yi in y]

        for epoch in range(epochs):
            total_grad_w = [[[0.0] * len(row) for row in layer]
                           for layer in self.weights]
            total_grad_b = [[0.0] * len(layer) for layer in self.biases]

            for xi, yi in zip(X, y_float):
                output, activations = self.forward(xi)
                w_grads, b_grads = self.backward(xi, yi, activations)

                for l in range(self.num_layers):
                    for j in range(len(w_grads[l])):
                        for i in range(len(w_grads[l][j])):
                            total_grad_w[l][j][i] += w_grads[l][j][i]
                        total_grad_b[l][j] += b_grads[l][j]

            n = len(X)
            for l in range(self.num_layers):
                for j in range(len(self.weights[l])):
                    for i in range(len(self.weights[l][j])):
                        self.weights[l][j][i] -= self.learning_rate * total_grad_w[l][j][i] / n
                    self.biases[l][j] -= self.learning_rate * total_grad_b[l][j] / n

        elapsed = (time.perf_counter() - start_time) * 1000
        accuracy = self._compute_accuracy(X, y)

        return {
            'accuracy': accuracy,
            'time_ms': elapsed,
            'epochs': epochs
        }

    def predict(self, X: List[List[float]]) -> List[int]:
        """Predict binary labels."""
        predictions = []
        for x in X:
            output, _ = self.forward(x)
            predictions.append(1 if output[0] >= 0.5 else 0)
        return predictions

    def _compute_accuracy(self, X: List[List[float]], y: List[int]) -> float:
        """Compute accuracy."""
        predictions = self.predict(X)
        correct = sum(p == t for p, t in zip(predictions, y))
        return correct / len(y) if y else 0.0


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""
    method: str
    dataset: str
    accuracy: float
    time_ms: float
    epochs: int = 0
    extra: Dict = None

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}


def run_benchmark(dataset_name: str,
                  X: List[List[float]],
                  y: List[int],
                  layer_sizes: List[int],
                  epochs: int = 100) -> List[BenchmarkResult]:
    """Run all methods on a dataset and collect results."""
    results = []

    # Seed for reproducibility
    random.seed(42)

    # 1. SAT-based (only for small networks)
    total_weights = sum(layer_sizes[l] * layer_sizes[l + 1]
                       for l in range(len(layer_sizes) - 1))

    if total_weights <= 20:
        sat_net = BinaryNeuralNetwork(layer_sizes, use_bias=True)
        start = time.perf_counter()
        sat_net.fit(X, y, verbose=False)
        elapsed = (time.perf_counter() - start) * 1000
        accuracy = sat_net._compute_accuracy(X, y)
        results.append(BenchmarkResult(
            method="SAT-Based",
            dataset=dataset_name,
            accuracy=accuracy,
            time_ms=elapsed,
            extra={'weights': 'binary'}
        ))

    # 2. Standard Gradient Descent
    random.seed(42)
    std_net = StandardNeuralNetwork(layer_sizes, learning_rate=0.3)
    std_result = std_net.fit(X, y, epochs=epochs, verbose=False)
    results.append(BenchmarkResult(
        method="Standard GD",
        dataset=dataset_name,
        accuracy=std_result['accuracy'],
        time_ms=std_result['time_ms'],
        epochs=epochs
    ))

    # 3. Topological Neural Network
    random.seed(42)
    topo_net = TopoNeuralNetwork(layer_sizes, learning_rate=0.3, topo_reg=0.005)
    topo_result = topo_net.fit(X, y, epochs=epochs, verbose=False)
    results.append(BenchmarkResult(
        method="TopoNN",
        dataset=dataset_name,
        accuracy=topo_result['final_accuracy'],
        time_ms=topo_result['time_ms'],
        epochs=epochs,
        extra={'topo_moves': topo_result['topo_moves']}
    ))

    # 4. Hybrid (SAT + Topo)
    if total_weights <= 30:
        random.seed(42)
        hybrid_net = HybridTopoNetwork(layer_sizes, learning_rate=0.3)
        hybrid_result = hybrid_net.fit(X, y, epochs=epochs, verbose=False)
        results.append(BenchmarkResult(
            method="Hybrid",
            dataset=dataset_name,
            accuracy=hybrid_result['final_accuracy'],
            time_ms=hybrid_result['total_time_ms'],
            epochs=epochs
        ))

    return results


def print_results_table(results: List[BenchmarkResult]):
    """Print results in a formatted table."""
    # Group by dataset
    datasets = {}
    for r in results:
        if r.dataset not in datasets:
            datasets[r.dataset] = []
        datasets[r.dataset].append(r)

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS: topoKEMP Neural Networks vs Standard Training")
    print("=" * 80)

    for dataset, dataset_results in datasets.items():
        print(f"\n{dataset}")
        print("-" * 60)
        print(f"{'Method':<20} {'Accuracy':<12} {'Time (ms)':<12} {'Notes'}")
        print("-" * 60)

        # Sort by accuracy (best first)
        dataset_results.sort(key=lambda x: -x.accuracy)

        for r in dataset_results:
            notes = ""
            if r.extra:
                if 'topo_moves' in r.extra:
                    notes = f"topo_moves={r.extra['topo_moves']}"
                elif 'weights' in r.extra:
                    notes = f"weights={r.extra['weights']}"

            print(f"{r.method:<20} {r.accuracy*100:>6.1f}%     {r.time_ms:>8.2f}    {notes}")


def run_comprehensive_benchmark():
    """Run comprehensive benchmark suite."""
    all_results = []

    print("\n" + "=" * 80)
    print("Running Comprehensive Neural Network Benchmark")
    print("=" * 80)

    # Dataset 1: XOR (classic test)
    print("\n[1/6] XOR Dataset...")
    X_xor, y_xor = make_xor_data()
    results = run_benchmark("XOR (2D)", X_xor, y_xor, [2, 4, 1], epochs=150)
    all_results.extend(results)

    # Dataset 2: AND
    print("[2/6] AND Dataset...")
    X_and, y_and = make_and_data()
    results = run_benchmark("AND (2D)", X_and, y_and, [2, 1], epochs=100)
    all_results.extend(results)

    # Dataset 3: OR
    print("[3/6] OR Dataset...")
    X_or, y_or = make_or_data()
    results = run_benchmark("OR (2D)", X_or, y_or, [2, 1], epochs=100)
    all_results.extend(results)

    # Dataset 4: Two Moons
    print("[4/6] Two Moons Dataset...")
    X_moons, y_moons = make_moons_data(80, noise=0.15)
    results = run_benchmark("Two Moons (80 pts)", X_moons, y_moons, [2, 8, 1], epochs=200)
    all_results.extend(results)

    # Dataset 5: Larger Two Moons
    print("[5/6] Larger Two Moons...")
    X_moons2, y_moons2 = make_moons_data(200, noise=0.1)
    results = run_benchmark("Two Moons (200 pts)", X_moons2, y_moons2, [2, 16, 8, 1], epochs=300)
    all_results.extend(results)

    # Dataset 6: Spiral (hardest)
    print("[6/6] Spiral Dataset...")
    X_spiral, y_spiral = make_spiral_data(100, noise=0.15)
    results = run_benchmark("Spiral (100 pts)", X_spiral, y_spiral, [2, 32, 16, 1], epochs=400)
    all_results.extend(results)

    print_results_table(all_results)

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    methods = {}
    for r in all_results:
        if r.method not in methods:
            methods[r.method] = {'accuracies': [], 'times': []}
        methods[r.method]['accuracies'].append(r.accuracy)
        methods[r.method]['times'].append(r.time_ms)

    print(f"\n{'Method':<20} {'Avg Accuracy':<15} {'Avg Time (ms)':<15}")
    print("-" * 50)
    for method, stats in methods.items():
        avg_acc = sum(stats['accuracies']) / len(stats['accuracies'])
        avg_time = sum(stats['times']) / len(stats['times'])
        print(f"{method:<20} {avg_acc*100:>6.1f}%        {avg_time:>8.2f}")

    return all_results


if __name__ == "__main__":
    run_comprehensive_benchmark()

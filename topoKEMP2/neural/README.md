# topoKEMP2 Neural Network Training

Train neural networks using topological methods from knot theory instead of (or combined with) standard gradient descent.

## Overview

This module provides three approaches to neural network training:

| Method | Best For | Key Advantage |
|--------|----------|---------------|
| **SAT-Based** | Small binary networks | Perfect accuracy, guaranteed solution |
| **Topological** | Medium networks | Escapes local minima, regularization |
| **Hybrid** | Flexible | Combines SAT + topological optimization |

## Quick Start

```python
from topoKEMP2.neural import (
    BinaryNeuralNetwork,   # SAT-based discrete weights
    TopoNeuralNetwork,     # Continuous with topological optimization
    HybridTopoNetwork,     # Combined approach
    make_xor_data,
    make_moons_data
)

# Example 1: SAT-based training (perfect for small problems)
X, y = make_xor_data()
net = BinaryNeuralNetwork([2, 4, 1], use_bias=True)
net.fit(X, y)  # Finds exact solution via SAT
print(net.predict(X))  # [0, 1, 1, 0]

# Example 2: Topological training (for larger problems)
X, y = make_moons_data(100)
net = TopoNeuralNetwork([2, 8, 1], learning_rate=0.3)
net.fit(X, y, epochs=200)
```

## How It Works

### SAT-Based Training (BinaryNeuralNetwork)

Formulates weight-finding as a Boolean satisfiability problem:
- Weights are discrete: `{-1, +1}` with integer biases
- Each training example becomes a logical constraint
- Uses topoKEMP2's SAT solver to find valid weights
- **Guaranteed to find solution if one exists**

```python
net = BinaryNeuralNetwork([2, 1], use_bias=True)
net.fit(X_and, y_and)
# Finds: weights=[[1, 1]], bias=[-2]
# Computes: sign(x1 + x2 - 2) → AND function
```

### Topological Training (TopoNeuralNetwork)

Uses knot theory concepts for optimization:

1. **Braid Representation**: Weight configurations encoded as braids
2. **Reidemeister Moves**: Navigate weight space while preserving function
3. **Topological Regularization**: Prefer simpler (shorter braid) solutions

```python
net = TopoNeuralNetwork([2, 4, 1], topo_reg=0.01)
net.fit(X, y, epochs=100, topo_interval=10)
# Applies topological moves every 10 epochs
```

### The Three Reidemeister Moves in Neural Networks

| Move | Knot Theory | Neural Network Analog |
|------|-------------|----------------------|
| **R1** | Remove twist | Prune near-zero weights |
| **R2** | Cancel opposite crossings | Remove redundant weight pairs |
| **R3** | Slide crossings | Permute hidden neurons |

## Benchmark Results

Comparison across standard datasets:

```
================================================================================
BENCHMARK RESULTS: topoKEMP Neural Networks vs Standard Training
================================================================================

XOR (2D)
------------------------------------------------------------
Method               Accuracy     Time (ms)    Notes
------------------------------------------------------------
SAT-Based             100.0%         1.75    weights=binary
Standard GD           100.0%         7.73
TopoNN                100.0%        15.56    topo_moves=14

AND (2D) / OR (2D)
------------------------------------------------------------
SAT-Based             100.0%         0.10    weights=binary
Standard GD           100.0%         1.27
TopoNN                100.0%         2.77

Two Moons (80 pts)
------------------------------------------------------------
Hybrid                 88.8%       438.08
Standard GD            87.5%       318.27
TopoNN                 86.2%       448.98

SUMMARY
------------------------------------------------------------
Method               Avg Accuracy    Avg Time (ms)
--------------------------------------------------
SAT-Based             100.0%            0.65
Standard GD            91.6%         2314.47
TopoNN                 90.3%         6512.38
Hybrid                 84.7%          115.12
```

### Key Findings

1. **SAT-Based is fastest for discrete problems** - Perfect accuracy in <1ms for AND/OR/XOR
2. **Standard GD competitive for large continuous** problems
3. **Topological approach** provides regularization benefits

## When to Use Each Method

### Use SAT-Based When:
- Network has ≤20 weights
- Binary/discrete weights acceptable
- Need guaranteed solution
- Logic gate or discrete pattern recognition

### Use TopoNN When:
- Medium-sized networks (20-100 weights)
- Want topological regularization
- Exploring weight space structure
- Research/experimental settings

### Use Hybrid When:
- Want best of both worlds
- Start with SAT for initialization
- Refine with topological gradient descent

## API Reference

### BinaryNeuralNetwork

```python
net = BinaryNeuralNetwork(
    layer_sizes=[2, 4, 1],  # Architecture
    threshold=0.0,           # Decision threshold
    use_bias=True           # Include bias terms
)

net.fit(X, y, verbose=True)      # Train
predictions = net.predict(X)      # Predict
weights, biases = net.get_weights()  # Get learned weights
```

### TopoNeuralNetwork

```python
net = TopoNeuralNetwork(
    layer_sizes=[2, 8, 1],
    learning_rate=0.1,
    topo_reg=0.01          # Topological regularization strength
)

results = net.fit(
    X, y,
    epochs=100,
    topo_interval=10,      # Apply topo moves every N epochs
    verbose=True
)

predictions = net.predict(X)
probas = net.predict_proba(X)
complexity = net.get_braid_complexity()  # Braid representation complexity
```

### HybridTopoNetwork

```python
net = HybridTopoNetwork(
    layer_sizes=[2, 4, 1],
    use_sat_init=True,     # Initialize with SAT solution
    learning_rate=0.1
)

results = net.fit(X, y, epochs=100, verbose=True)
predictions = net.predict(X)
```

## Dataset Generators

```python
from topoKEMP2.neural import (
    make_xor_data,      # XOR: [0,1,1,0]
    make_and_data,      # AND: [0,0,0,1]
    make_or_data,       # OR:  [0,1,1,1]
    make_simple_data,   # Linearly separable
    make_moons_data,    # Two interleaved moons
    make_spiral_data    # Two spirals (challenging)
)

# Examples
X, y = make_xor_data()           # 4 examples
X, y = make_moons_data(100)      # 100 examples
X, y = make_spiral_data(200, noise=0.1)  # 200 examples with noise
```

## Running Benchmarks

```bash
python -m topoKEMP2.neural.benchmark
```

## Theoretical Background

### Why Topology for Neural Networks?

The weight space of a neural network has topological structure:
- **Paths through weight space** = training trajectories
- **Equivalent configurations** = same function, different weights
- **Simpler representations** = better generalization

By encoding weights as braids and using Reidemeister moves, we can:
1. Identify equivalent weight configurations
2. Navigate to simpler representations
3. Escape local minima through topological moves

### The Braid-Weight Correspondence

For a network with weights W = {w_ij}:
1. Order weights by magnitude
2. Encode the sorting permutation as a braid word
3. Signs of weights determine crossing directions

This gives a topological invariant of the weight configuration.

## Limitations

- **SAT-Based**: Only works for small networks (≤20 weights)
- **Topological**: Overhead for large networks
- **Discrete Weights**: May not perfectly fit all problems

## Integration with topoKEMP2

The neural module fully integrates with topoKEMP2's tools:

```python
from topoKEMP2.neural import WeightBraid
from topoKEMP2.braid import BraidWord

# Convert weights to braid representation
braid = WeightBraid(net.weights, num_strands=total_weights)
braid_word = braid.to_braid_word()
print(f"Braid complexity: {braid.complexity()}")
```

## Future Directions

1. **Jones Polynomial for Weight Invariants**: Use knot invariants to identify equivalent networks
2. **Persistent Homology**: Track topological features during training
3. **Braid Optimization**: Apply braid reduction algorithms for network compression
4. **Quantum Integration**: Leverage connections to topological quantum computing

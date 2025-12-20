"""
topoKEMP2 Neural Network Module

Train neural networks using SAT solving instead of gradient descent.

QUICK START:
    from topoKEMP2.neural import SATNeuralNetwork, train_sat

    # Train on XOR
    X = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
    y = [0, 1, 1, 0]

    net = train_sat(X, y, hidden_sizes=[4])
    predictions = net.predict(X)

KEY CONCEPTS:
- Weights are discretized to finite values (e.g., {-1, 0, 1})
- Training = finding weight assignments that classify correctly
- Uses topoKEMP2 SAT solvers for optimization

ADVANTAGES:
- Complete solver (finds solution if exists)
- No local minima
- Interpretable discrete weights

LIMITATIONS:
- Works best for small networks
- Requires weight quantization
"""

from .sat_nn import (
    SATNeuralNetwork,
    BinaryNeuralNetwork,
    train_sat,
    make_xor_data,
    make_and_data,
    make_or_data,
    make_simple_data,
)

__all__ = [
    "SATNeuralNetwork",
    "BinaryNeuralNetwork",
    "train_sat",
    "make_xor_data",
    "make_and_data",
    "make_or_data",
    "make_simple_data",
]

# NeuroGrad

A micrograd-like autodiff library extended to support vector operations. NeuroGrad provides automatic differentiation capabilities for building and training neural networks from scratch.

## Features

- **Vector-based automatic differentiation**: Extends micrograd's scalar operations to vectors
- **Neural network components**: Built-in `Neuron`, `Layer`, and `MLP` classes
- **Activation functions**: ReLU and Tanh activations with automatic gradient computation
- **Simple API**: Easy-to-use interface for building neural networks

## Installation

Clone the repository and install:

```bash
git clone https://github.com/Inomjonov/neurograd.git
cd neurograd
pip install -e .
```

Or install directly from GitHub:

```bash
pip install git+https://github.com/Inomjonov/neurograd.git
```

## Quick Start

```python
from neurograd import VectorValue, Neuron, Layer, MLP

# Create a simple neural network
mlp = MLP(3, [4, 4, 1])

# Forward pass
x = VectorValue([1.0, 2.0, 3.0])
out = mlp(x)

# Backward pass
loss = (out - VectorValue([1.0])) ** 2
loss.backward()

# Access gradients
for p in mlp.parameters():
    print(f"Parameter: {p.data}, Gradient: {p.grad}")
```

## Components

### VectorValue

The core class for automatic differentiation with vector support.

```python
from neurograd import VectorValue

a = VectorValue([1.0, 2.0, 3.0])
b = VectorValue([4.0, 5.0, 6.0])

# Operations
c = a + b
d = a * b
e = a.dot(b)
f = a.sum()
```

### Neuron

A single neuron with optional non-linearity.

```python
from neurograd import Neuron

neuron = Neuron(nin=3, nonlin=True)
output = neuron(VectorValue([1.0, 2.0, 3.0]))
```

### Layer

A layer of neurons.

```python
from neurograd import Layer

layer = Layer(nin=3, nout=4)
output = layer(VectorValue([1.0, 2.0, 3.0]))
```

### MLP

Multi-layer perceptron.

```python
from neurograd import MLP

mlp = MLP(nin=3, nouts=[4, 4, 1])
output = mlp(VectorValue([1.0, 2.0, 3.0]))
```

## License

See LICENSE file for details.

## Author

Mironshoh Inomjonov

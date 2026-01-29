from neurograd.neurograd.engine import VectorValue
import random

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = [0.0 for _ in p.grad]
    
    def parameters(self):
        return []
    
class Neuron(Module):
    def __init__(self, nin, nonlin = True):
        self.w = VectorValue(
            [random.uniform(-1, 1) for _ in range(nin)], 
             label='w'
        )
    
        self.b = VectorValue([0.0], label='b')
        self.nonlin = nonlin

    def __call__(self, x):
        x = x if isinstance(x, VectorValue) else VectorValue(x)
        act = (self.w.dot(x)) + self.b
        return act.tanh() if self.nonlin else act
    
    def parameters(self):
        return [self.w, self.b]
    
    def __repr__(self):
        return f"{'Tanh' if self.nonlin else 'Linear'} Neuron ({len(self.w.data)})"
     

class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        x = x if isinstance(x, VectorValue) else VectorValue(x)
        values = [n(x).data[0] for n in self.neurons]
        return VectorValue(values)


    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
    
class MLP(Module):
    def __init__(self, nin, nouts):
        size = [nin] + nouts
        self.layers = [
            Layer(
                size[i],
                size[i+1],
                nonlin = i != len(nouts) -1
                )
            for i in range(len(nouts))
        ]
    
    def __call__(self, x):
        x = x if isinstance(x, VectorValue) else VectorValue(x)
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"


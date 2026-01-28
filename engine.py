import math

class VectorValue:
    def __init__(self, data, _children = (), _op: str = "", label: str = ""):
        if isinstance(data, (int, float)):
            self.data = [float(data)]
        else:
            self.data = list(data)
            
        self.grad = [0.0 for _ in self.data]
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"VectorValue(data = {self.data})"
    
    # Addition(s)
    def __add__(self, other):
        other = other if isinstance(other, VectorValue) else VectorValue(other)
        assert len(self.data) == len(other.data)

        out_data = [a + b for a, b in zip(self.data, other.data)]
        out = VectorValue(out_data, (self, other), "+")

        def _backward():
            for i in range(len(self.grad)):
                self.grad[i] += out.grad[i]
                other.grad[i] += out.grad[i]
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        # Handle sum() starting with 0: convert other to VectorValue
        other = other if isinstance(other, VectorValue) else VectorValue(other)
        return self + other
    

    # Multiplications
    def __mul__(self, other):
        other = other if isinstance(other, VectorValue) else VectorValue(other)
        assert len(other.data) == len(self.data)
        out_data = [a * b for a, b in zip(self.data, other.data)]
        out = VectorValue(out_data, (self, other), "*")

        def _backward():
            for i in range(len(self.grad)):
                self.grad[i] += other.data[i] * out.grad[i]
                other.grad[i] += self.data[i] * out.grad[i]
        
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    # Dot product
    def dot(self, other):
        assert len(self.data) == len(other.data)
        out_data = [sum(a * b for a, b in zip(self.data, other.data))]
        out = VectorValue(out_data, (self, other), "dot-product")

        def _backward():
            for i in range(len(self.grad)):
                self.grad[i] += other.data[i] * out.grad[0]
                other.grad[i] += self.data[i] * out.grad[0]
        
        out._backward = _backward
        return out  
    
    # Sum
    def sum(self):
        out_data = [sum(self.data)]
        out = VectorValue(out_data, (self, ), "sum")

        def _backward():
            for i in range(len(self.data)):
                self.grad[i] += out.grad[0]
        
        out._backward = _backward
        return out
    
    # Activations

    def relu(self):
        out_data = [max(0.0, x) for x in self.data]
        out = VectorValue(out_data, (self, ), "ReLU")

        def _backward():
            for i in range(len(self.data)):
                self.grad[i] += (self.data[i] > 0) * out.grad[i]
        
        out._backward = _backward
        return out
    
    def tanh(self):
        out_data = [math.tanh(x) for x in self.data]
        out = VectorValue(out_data, (self,), "tanh")

        def _backward():
            for i in range(len(self.data)):
                self.grad[i] += (1 - out.data[i]**2) * out.grad[i]
        
        out._backward = _backward
        return out
             
    
    # Negation
    def __neg__(self):
        return self * -1
    
    # Subtractions
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __pow__(self, power):
        assert isinstance(power, (int, float)), "only supports scalar powers"

        out_data = [x ** power for x in self.data]
        out = VectorValue(out_data, (self,), f"**{power}")

        def _backward():
            for i in range(len(self.data)):
                self.grad[i] += power * (self.data[i] ** (power - 1)) * out.grad[i]

        out._backward = _backward
        return out

        
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if not v in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = [1.0 for _ in self.data]
        for v in reversed(topo):
            v._backward()

    
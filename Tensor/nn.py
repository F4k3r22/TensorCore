from .tensor import Tensor
import math

class Linear:
    def __init__(self, in_features, out_features, requires_grad=True):
        self.requires_grad = requires_grad
        
        # Crear el tensor de entrada
        input_tensor = Tensor.tensor((in_features, out_features))
        scale_factor = math.sqrt(2. / in_features)
        
        # Escalar los datos del tensor de entrada
        scaled_data = [[val * scale_factor for val in row] 
                      for row in input_tensor.data]
        
        # Crear los tensores W y b
        self.W = Tensor(scaled_data, requires_grad=True)
        self.b = Tensor([[1.0 for _ in range(out_features)]], requires_grad=True)
        
        self.W._ensure_grad()
        self.b._ensure_grad()

    def __call__(self, x):
        out = x.matmul(self.W) + self.b
        return out
    
    def parameters(self):
        return [self.W, self.b]
    
    def zero_grad(self):
        self.W.grad = Tensor.zeros(self.W.data)
        self.b.grad = Tensor.zeros(self.b.data)
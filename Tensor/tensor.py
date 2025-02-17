import random
from typing import List, Tuple, Union
import math
from .utils import *

def CreateMatrix(size: tuple, range_: tuple) -> List[List[int]]:
    """
    Args:
        size: Es el tamaño de la matriz, por ej. puede ser [3,3] que seria una matriz de 3x3
        range_: Es el rango de valores que puede tener, por ej. [1,190] que cubre desde el 1 hasta el 190 + 1
    Returns:
        List[List[int]]: Matriz generada con valores secuenciales
    """
    size_a, size_b = size
    range_a, range_b = range_
    rangenumber = random.randint(range_a, range_b)
    return [[rangenumber + i * size_b + j for j in range(size_b)] for i in range(size_a)]

def SumMatrix(a: List[List[int]], b: List[List[int]]) -> Union[str, List[List[int]]]:
    """
    Suma de Matrices
    Args:
        a: Matriz a que quieras sumar
        b: Matriz b que se va a sumar con la a
    Returns:
        Union[str, List[List[int]]]: Resultado de la suma o mensaje de error
    """
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        return "Las matrices deben tener las mismas dimensiones"
    
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]

class Tensor:
    def __init__(self, data: Union[List, Tuple], requires_grad=False):
        """
        Convierte la matriz/lista a un Tensor
        Args:
            data: Datos para crear el tensor
        """
        self.data = data
        self.shape = self._get_shape(data)
        self.rank = len(self.shape)
        self.requires_grad = requires_grad
        self.grad = Tensor.zeros(self.data) if requires_grad else None
        
    def _get_shape(self, data: Union[List, Tuple]) -> Tuple:
        """
        Calcula recursivamente la forma del tensor
        Args:
            data: Datos para calcular la forma
        Returns:
            Tuple: Forma del tensor
        """
        shape = []
        current = data
        while isinstance(current, (list, tuple)):
            shape.append(len(current))
            if len(current) > 0:
                current = current[0]
            else:
                break
        return tuple(shape)
    
    def __str__(self) -> str:
        return f"Tensor(data={self.data}, shape={self.shape}, rank={self.rank})"
    
    def __getitem__(self, index):
        return self.data[index]

    def _ensure_grad(self):
        if self.grad is None:
            self.grad = Tensor.zeros(self.data)
    
    def transpose(self) -> 'Tensor':
        """
        Transpone el tensor (solo para tensores 2D)
        Returns:
            Tensor: Tensor transpuesto
        """
        if self.rank != 2:
            raise ValueError("Transpose solo está implementado para tensores 2D")
        transposed = [[self.data[j][i] for j in range(self.shape[0])] 
                     for i in range(self.shape[1])]
        return Tensor(transposed)

    def backward(self):
        self._ensure_grad()

        topo = []
        visited = set()

        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for parent in t._prev:
                    build_topo(parent)
                topo.append(t)
        
        build_topo(self)

        self.grad = Tensor.ones_like(self.data)

        for t in reversed(topo):
            t._backward()

    def __add__(self, other):

        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():

            def apply_grad(tensor, grad):
                
                if isscalar(grad) or self._get_shape(grad) == 1:
                    tensor.grad += Tensor.sum(grad)
                else:
                    while tensor.grad.ndim < grad.ndim:
                        tensor.grad = Tensor.expand_dims(tensor.grad, axis=-1)
                    while grad.ndim < tensor.grad.ndim:
                        grad = Tensor.expand_dims(grad, axis=-1)

                    if tensor.grad._get_shape != grad._get_shape:

                        axes = tuple(i for i in range(grad.ndim) if tensor.grad._get_shape[i] == 1 and grad._get_shape[i] != 1)
                        grad = Tensor.sum(grad, axis=axes)
                    
                    tensor.grad += grad
                
                if self.requires_grad:
                    self._ensure_grad()
                    grad_other = out.grad

                    apply_grad(self, grad_other)
            
            out._backward = _backward
            out._prev = [self, other]
            out._op = '+'

            return out

    def multiply(self, other: 'Tensor') -> 'Tensor':
        """
        Multiplicación elemento a elemento
        Args:
            other: Otro tensor para multiplicar
        Returns:
            Tensor: Resultado de la multiplicación
        """
        if self.rank != 2 or other.rank != 2:
            raise ValueError("Dot product solo está implementado para tensores 2D")
        
        if self.shape[1] != other.shape[0]:
            raise ValueError(f"Formas incompatibles para dot product: {self.shape} y {other.shape}")
        
        result = []
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
            # Multiplicar fila i de self por columna j de other
                element = sum(self.data[i][k] * other.data[k][j] 
                            for k in range(self.shape[1]))
                row.append(element)
            result.append(row)
    
        return Tensor(result)
        
    
    def dot(self, other: 'Tensor') -> 'Tensor':
        """
        Realiza la multiplicación matricial entre dos tensores
        Args:
            other: Otro tensor para multiplicar
        Returns:
            Tensor: Resultado de la multiplicación matricial
        """
        if self.rank != 2 or other.rank != 2:
            raise ValueError("Dot product solo está implementado para tensores 2D")
        
        if self.shape[1] != other.shape[0]:
            raise ValueError(f"Formas incompatibles para dot product: {self.shape} y {other.shape}")
    
        result = []
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
                element = sum(self.data[i][k] * other.data[k][j] 
                            for k in range(self.shape[1]))
                row.append(element)
            result.append(row)
    
        return Tensor(result)

    def sum(self, axis=None):

        if axis is None:
            if self.rank == 1:
                return Tensor(sum(self.data))
            elif self.rank == 2:
                return Tensor(sum(sum(row) for row in self.data))
            else: 
                raise NotImplementedError("Sum sin eje solo implementado para tensores 1D y 2D")
        
        if not isinstance(axis, int):
            raise TypeError("El eje debe ser un entero")
        
        if self.rank == 1:
            if axis == 0:
                return Tensor(sum(self.data))
            raise ValueError("Tensor 1D solo acepta axis=0 o None")
        
        elif self.rank == 2:
            if axis == 0:
                return Tensor([sum(col) for col in zip(*self.data)])
            elif axis == 1:
                return Tensor([sum(row) for row in self.data])
            raise ValueError("Tensor 2D solo acepta axis=0, axis=1 o None")
        
        else:
            raise NotImplementedError("Sum solo implementado para tensores 1D y 2D")
        
    def expand_dims(self, axis=0):

        if axis < 0:
            axis = self.rank + 1 + axis

        if not -1 <= axis <= self.rank:
            raise ValueError(f"Eje {axis} está fuera de rango para tensor de rango {self.rank}")
        
        def wrap_data(data, current_depth, target_depth):
            if current_depth == target_depth:
                return [data]
            if isinstance(data, (list, tuple)):
                return [wrap_data(d, current_depth, target_depth) for d in data]
            return [data]
        
        expanded_data = wrap_data(self.data, 0, axis)
        return Tensor(expanded_data)
    
    @staticmethod
    def zeros(data):
        """
        Crea un tensor lleno de ceros con la misma forma que los datos de entrada
        Args:
            data: Datos de referencia para obtener la forma
        Returns:
            Tensor: Tensor lleno de ceros
        """
        def get_shape(data):
            shape = []
            current = data
            while isinstance(current, (list, tuple)):
                shape.append(len(current))
                if len(current) > 0:
                    current = current[0]
                else:
                    break
            return tuple(shape)

        def create_zeros(dims):
            if len(dims) == 1:
                return [0] * dims[0]
            return [create_zeros(dims[1:]) for _ in range(dims[0])]
        
        shape = get_shape(data)
        return Tensor(create_zeros(shape))

    @staticmethod
    def ones_like(data):
        """
        Crea un tensor lleno de unos con la misma forma que los datos de entrada
        Args:
            data: Datos de referencia para obtener la forma
        Returns:
            Tensor: Tensor lleno de unos
        """
        def get_shape(data):
            shape = []
            current = data
            while isinstance(current, (list, tuple)):
                shape.append(len(current))
                if len(current) > 0:
                    current = current[0]
                else:
                    break
            return tuple(shape)

        def create_ones(dims):
            if len(dims) == 1:
                return [1.0] * dims[0]
            return [create_ones(dims[1:]) for _ in range(dims[0])]
    
        shape = get_shape(data)
        return Tensor(create_ones(shape))
    
    @staticmethod
    def tensor(shape: Tuple[int, ...]) -> 'Tensor':
        """
        Crea un tensor 
        Args:
            shape: Forma deseada del tensor
        Returns:
            Tensor: Tensor lleno de numeros tipo float
        """
        def create_tensor(dims):
            if len(dims) == 1:
                return [random.random() for _ in range(dims[0])]
            return [create_tensor(dims[1:]) for _ in range(dims[0])]
    
        return Tensor(create_tensor(shape))
    
    @property
    def ndim(self):
        return self.rank


class Layer:
    def __init__(self, input_size: int, output_size: int):
        """
        Capa fully connected de la red neuronal
        
        Args:
            input_size: Número de entradas
            output_size: Número de neuronas en la capa
        """
        # Inicializar pesos y biases con valores aleatorios pequeños
        self.weights = Tensor([[random.uniform(-0.01, 0.01) for _ in range(input_size)] 
                             for _ in range(output_size)], requires_grad=True)
        self.bias = Tensor([[random.uniform(-0.01, 0.01)] for _ in range(output_size)], 
                          requires_grad=True)
        
    def forward(self, x: Tensor) -> Tensor:
        # Asegurarnos que x sea 2D
        if x.rank == 1:
            x = Tensor([x.data])  # Convertir a 2D
            
        # y = wx + b
        return self.weights.dot(x.transpose()).transpose() + self.bias

def sigmoid(x: Tensor) -> Tensor:
    """Función de activación sigmoid"""
    def sigmoid_scalar(x: float) -> float:
        # Clipear valores para evitar overflow
        x = max(min(x, 20), -20)
        return 1 / (1 + math.exp(-x))
    
    if isinstance(x.data[0], list):  # Para tensores 2D
        return Tensor([[sigmoid_scalar(val) for val in row] 
                      for row in x.data])
    else:  # Para tensores 1D
        return Tensor([sigmoid_scalar(val) for val in x.data])

def binary_cross_entropy(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """Loss function: Binary Cross Entropy"""
    epsilon = 1e-15  # Para evitar log(0)
    total_loss = 0
    n = 0
    
    # Calculamos la pérdida para cada par de predicción/label
    for pred_row, true_row in zip(y_pred.data, y_true.data):
        for pred, true in zip(pred_row, true_row):
            # Clipear valores para evitar log(0)
            pred = max(min(pred, 1 - epsilon), epsilon)
            total_loss += true * math.log(pred) + (1 - true) * math.log(1 - pred)
            n += 1
            
    return Tensor(-total_loss / n)

class NeuralNetwork:
    def __init__(self, layer_sizes: List[int]):
        """
        Red neuronal simple feed-forward
        
        Args:
            layer_sizes: Lista con el tamaño de cada capa, incluyendo entrada y salida
        """
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1]))
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass"""
        activation = x
        for layer in self.layers:
            activation = sigmoid(layer.forward(activation))
        return activation
    
    def train_step(self, x: Tensor, y: Tensor, learning_rate: float = 0.01) -> float:
        """
        Realiza un paso de entrenamiento
        
        Args:
            x: Datos de entrada
            y: Labels verdaderos
            learning_rate: Tasa de aprendizaje
            
        Returns:
            float: Valor de la pérdida en este paso
        """
        # Forward pass
        y_pred = self.forward(x)
        
        # Calcular pérdida
        loss = binary_cross_entropy(y_pred, y)
        
        # Backward pass
        loss.backward()
        
        # Actualizar pesos
        for layer in self.layers:
            # Actualizar pesos
            for i in range(len(layer.weights.data)):
                for j in range(len(layer.weights.data[i])):
                    layer.weights.data[i][j] -= learning_rate * layer.weights.grad.data[i][j]
            
            # Actualizar biases
            for i in range(len(layer.bias.data)):
                layer.bias.data[i][0] -= learning_rate * layer.bias.grad.data[i][0]
            
            # Resetear gradientes
            layer.weights.grad = None
            layer.bias.grad = None
            
        return loss.data
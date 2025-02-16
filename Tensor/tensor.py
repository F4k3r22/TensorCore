import random
from typing import List, Tuple, Union
import math

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
            pass

    def __add__(self, other):

        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():

            def apply_grad(tensor, grad):
                pass
    
    def multiply(self, other: 'Tensor') -> 'Tensor':
        """
        Multiplicación elemento a elemento
        Args:
            other: Otro tensor para multiplicar
        Returns:
            Tensor: Resultado de la multiplicación
        """
        if self.shape != other.shape:
            raise ValueError("Las formas deben coincidir para multiplicación elemento a elemento")
        
        if self.rank == 2:
            result = [[self.data[i][j] * other.data[i][j] 
                      for j in range(self.shape[1])] 
                     for i in range(self.shape[0])]
            return Tensor(result)
        raise NotImplementedError("Multiplicación solo implementada para tensores 2D")
    
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

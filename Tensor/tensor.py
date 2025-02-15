import random

def CreateMatrix(size: tuple, range_: tuple):
    """
    Args:
        size: Es el tamaño de la matriz, por ej. puede ser [3,3] que seria una matriz de 3x3
        range_: Es el rango de valores que puede tener, por ej. [1,190] que cubre desde el 1 hasta el 190 + 1
    """
    # size_a es una fila
    # size_b es una columna
    size_a, size_b = size
    # range_a y range_b son el rango de numeros que va a contener la matriz 
    # es decir desde el numero a hasta el numero b + 1
    range_a, range_b = range_
    rangenumber = random.randint(range_a, range_b)
    matrix = []
    for i in range(size_a):
        a = []
        for j in range(size_b):
            a.append(rangenumber)
            rangenumber = rangenumber + 1
        matrix.append(a)
    return matrix

def SumMatrix(a: list, b: list):
    """
    Suma de Matrices:
    Args:
        a: Matriz a que quieras sumar
        b: Matriz b que se va a sumar con la a
    """
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        return "Las matrices deben tener las mismas dimensiones"
    
    result = []
    for i in range(len(a)):
        row = []
        for j in range(len(a[0])):
            # Sumar los elementos correspondientes
            row.append(a[i][j] + b[i][j])
        result.append(row)
    
    return result

class TensorConvert:
    def __init__(self, data):
        """ Conviertes la matriz a un Tensor"""
        self.data = data
        self.shape = self._get_shape(data)
        self.rank = len(self.shape)
        
    def _get_shape(self, data):
        """ Calcula recursivamente la forma del tensor"""
        shape = []
        current = data

        while isinstance(current, (list, tuple)):
            shape.append(len(current))
            if len(current) > 0:
                current = current[0]
            else:
                break

        return tuple(shape)
    
    def __str__(self):
        return f"Tensor(data={self.data}, shape={self.shape}, rank={self.rank})"
    
    def __getitem__(self, index):
        """ Acceder a elementos usando corchetes"""
        return self.data[index]
    

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

def softmax(tensor):
    if tensor.rank == 1:
        # Handle 1D case
        data_exp = [math.exp(x) for x in tensor.data]
        sum_exp = sum(data_exp)
        result = [x / sum_exp for x in data_exp]
        return Tensor(result)
    elif tensor.rank == 2:
        # Handle 2D case
        result = []
        for row in tensor.data:
            # Apply softmax to each row
            row_exp = [math.exp(x) for x in row]
            row_sum = sum(row_exp)
            row_softmax = [x / row_sum for x in row_exp]
            result.append(row_softmax)
        return Tensor(result)
    else:
        raise ValueError(f"Softmax not implemented for tensors of rank {tensor.rank}")
    
def TensorNormalize(X):
    """
    Normaliza un tensor 2D usando la fórmula (x - mean) / (std + epsilon)
    Args:
        X: Lista de listas de Tensores
    Returns:
        Lista de listas de Tensores normalizados
    """
    # Primero convertimos los Tensores a una matriz de números
    X_data = [[x.data for x in row] for row in X]
    
    # Calculamos la media por columna
    def column_mean(matrix, col_idx):
        col_sum = sum(row[col_idx] for row in matrix)
        return col_sum / len(matrix)
    
    # Calculamos la desviación estándar por columna
    def column_std(matrix, col_idx, mean):
        squared_diff_sum = sum((row[col_idx] - mean) ** 2 for row in matrix)
        return (squared_diff_sum / len(matrix)) ** 0.5
    
    # Obtenemos número de filas y columnas
    n_rows = len(X_data)
    n_cols = len(X_data[0])
    
    # Calculamos media y desviación estándar para cada columna
    means = [column_mean(X_data, j) for j in range(n_cols)]
    stds = [column_std(X_data, j, means[j]) for j in range(n_cols)]
    
    # Normalizamos los datos
    epsilon = 1e-8
    X_norm = []
    for i in range(n_rows):
        row_norm = []
        for j in range(n_cols):
            normalized_value = (X_data[i][j] - means[j]) / (stds[j] + epsilon)
            row_norm.append(Tensor(normalized_value, requires_grad=False))
        X_norm.append(row_norm)
    
    return X_norm
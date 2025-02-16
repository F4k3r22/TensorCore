from Tensor import Tensor
from Tensor.nn import softmax

a = Tensor.tensor((1, 3))
print("Tensor a:")
print(a)


a_soft = softmax(a)
print("Tensor a despues de aplicar softmax:")
print(a_soft)
from Tensor import CreateMatrix, Tensor

a = CreateMatrix([3,3],[-100,900])
b = CreateMatrix([3,3],[-100, 120])

tensor_a = Tensor(a)
tensor_b = Tensor(b)

print("Tensor a:")
print(tensor_a)

print("Tensor b:")
print(tensor_b)

print("\nTranspuesta de A:")
print(tensor_a.transpose())

print("\nMultiplicación elemento a elemento:")
print(tensor_a.multiply(tensor_b))
    
print("\nTensor de ceros 2x3x2:")
zeros = Tensor.zeros((2, 3, 2))
print(zeros)

print("\nTensor de 1x768")
tensor = Tensor.tensor((1,768))
print(tensor)
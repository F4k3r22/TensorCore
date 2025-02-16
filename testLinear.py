from Tensor.nn import Linear

print("Test de la clase Linear")
linear = Linear(4,3)
print(f"Dimensiones de W: {linear.W.shape}")
print(f"Dimensiones de b: {linear.b.shape}")
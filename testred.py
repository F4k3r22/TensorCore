from Tensor import *

X = Tensor([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    
y = Tensor([
        [0],
        [1],
        [1],
        [0]
    ])
    
# Crear red neuronal: 2 -> 4 -> 1
model = NeuralNetwork([2, 4, 1])
    
# Entrenamiento
epochs = 100
for epoch in range(epochs):
    loss = model.train_step(X, y)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
# Predicciones
predictions = model.forward(X)
print("\nPredicciones finales:")
for x_i, pred in zip(X.data, predictions.data):
    print(f"Input: {x_i}, Prediction: {pred[0]:.4f}")
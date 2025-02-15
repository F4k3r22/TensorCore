from Tensor import CreateMatrix, SumMatrix

a = CreateMatrix([3,3],[1,900])
b = CreateMatrix([3,3],[1, 120])

print(f"Matriz 1:")
for i in a:
    print(i)

print(f"Matriz 2:")
for j in b:
    print(j)

suma = SumMatrix(a, b)

print(f"Suma de las 2 matrices")
for c in suma:
    print(c)
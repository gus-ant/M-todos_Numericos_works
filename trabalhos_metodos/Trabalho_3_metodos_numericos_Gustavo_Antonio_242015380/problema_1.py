# PROBLEMA 1 
# Jacobi e Gauss-Seidel para sistema 10x10

import numpy as np

# Dados do problema
b = np.array([-110, -30, -40, -110, 0, -15, -90, -25, -55, -65], dtype=float)
x0 = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0])
n = len(b)

# Criar matriz A diagonal dominante (exemplo)
np.random.seed(42)
A = np.random.rand(n, n) * 5  # valores entre 0 e 5
for i in range(n):
    A[i, i] = np.sum(np.abs(A[i, :])) + 10  # garantir diagonal dominante

print("Matriz A (diagonal dominante):")
print(A)
print("\nVetor b:")
print(b)

# Método de Jacobi
def jacobi(A, b, x0, tol=1e-6, max_iter=1000):
    n = len(b)
    x = x0.copy()
    x_new = np.zeros_like(x)
    for k in range(max_iter):
        for i in range(n):
            soma = 0.0
            for j in range(n):
                if j != i:
                    soma += A[i, j] * x[j]
            x_new[i] = (b[i] - soma) / A[i, i]
        erro = np.linalg.norm(x_new - x) / np.linalg.norm(x)
        x = x_new.copy()
        if erro < tol:
            print(f"Jacobi convergiu em {k+1} iterações")
            return x
    print("Jacobi não convergiu no número máximo de iterações")
    return x

# Método de Gauss-Seidel
def gauss_seidel(A, b, x0, tol=1e-6, max_iter=1000):
    n = len(b)
    x = x0.copy()
    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            soma1 = np.dot(A[i, :i], x[:i])
            soma2 = np.dot(A[i, i+1:], x_old[i+1:])
            x[i] = (b[i] - soma1 - soma2) / A[i, i]
        erro = np.linalg.norm(x - x_old) / np.linalg.norm(x_old)
        if erro < tol:
            print(f"Gauss-Seidel convergiu em {k+1} iterações")
            return x
    print("Gauss-Seidel não convergiu no número máximo de iterações")
    return x

# Resolver
print("\n" + "="*50)
print("SOLUÇÃO PELO MÉTODO DE JACOBI:")
x_jacobi = jacobi(A, b, x0)
print(f"Solução: {x_jacobi}")

print("\n" + "="*50)
print("SOLUÇÃO PELO MÉTODO DE GAUSS-SEIDEL:")
x_gs = gauss_seidel(A, b, x0)
print(f"Solução: {x_gs}")

# Verificação
print("\n" + "="*50)
print("VERIFICAÇÃO (A*x - b):")
print("Para Jacobi:   ", np.dot(A, x_jacobi) - b)
print("Para Gauss-Seidel:", np.dot(A, x_gs) - b)
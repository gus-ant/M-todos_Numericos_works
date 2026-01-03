# PROBLEMA 2 
# Eliminação de Gauss e Decomposição LU (Crout)

import numpy as np

np.random.seed(123)
n = 17

# Criar matriz aleatória e vetor b
A = np.random.rand(n, n) * 10 - 5  # valores entre -5 e 5
b = np.random.rand(n) * 20 - 10    # valores entre -10 e 10

# Garantir que não há zeros na diagonal (para pivoteamento)
for i in range(n):
    if abs(A[i, i]) < 0.1:
        A[i, i] = 1.0

print("Matriz A (17x17):")
print(A)
print("\nVetor b:")
print(b)

# ELIMINAÇÃO DE GAUSS COM PIVOTEAMENTO PARCIAL
def gauss_elimination(A, b):
    n = len(b)
    Ab = np.hstack([A.astype(float), b.reshape(-1, 1)])  # matriz aumentada
    
    for i in range(n):
        # Pivoteamento parcial
        max_row = np.argmax(np.abs(Ab[i:, i])) + i
        Ab[[i, max_row]] = Ab[[max_row, i]]
        
        if abs(Ab[i, i]) < 1e-12:
            raise ValueError("Matriz singular")
        
        # Eliminação
        for j in range(i+1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]
    
    # Substituição regressiva
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:])) / Ab[i, i]
    
    return x

# DECOMPOSIÇÃO LU PELO MÉTODO DE CROUT
def crout_lu(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    
    for i in range(n):
        U[i, i] = 1.0  # Diagonal de U é 1
    
    for j in range(n):
        for i in range(j, n):
            soma = 0.0
            for k in range(j):
                soma += L[i, k] * U[k, j]
            L[i, j] = A[i, j] - soma
        
        for i in range(j+1, n):
            soma = 0.0
            for k in range(j):
                soma += L[j, k] * U[k, i]
            U[j, i] = (A[j, i] - soma) / L[j, j]
    
    return L, U

def solve_lu(L, U, b):
    # Ly = b
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
        y[i] /= L[i, i]
    
    # Ux = y
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = y[i] - np.dot(U[i, i+1:], x[i+1:])
    
    return x

print("\n" + "="*60)
print("SOLUÇÃO POR ELIMINAÇÃO DE GAUSS COM PIVOTEAMENTO:")
x_gauss = gauss_elimination(A.copy(), b.copy())
print(f"Solução: {x_gauss}")

print("\n" + "="*60)
print("SOLUÇÃO POR DECOMPOSIÇÃO LU (CROUT):")
L, U = crout_lu(A.copy())
x_lu = solve_lu(L, U, b.copy())
print(f"Solução: {x_lu}")

# Verificação
print("\n" + "="*60)
print("VERIFICAÇÃO:")
print(f"Erro Gauss:    {np.linalg.norm(np.dot(A, x_gauss) - b):.2e}")
print(f"Erro LU:       {np.linalg.norm(np.dot(A, x_lu) - b):.2e}")

# Verificar se LU = A
print(f"\nDiferença entre A e L*U: {np.linalg.norm(A - np.dot(L, U)):.2e}")
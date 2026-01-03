import numpy as np
import matplotlib.pyplot as plt

# Parâmetros
L = 4.0
EI = 1.4e7
q = 10e3
h = 0.01  # passo
tolerance = 1e-5

# Sistema de EDOs
def f(x, y):
    dydx0 = y[1]
    dydx1 = ( (1 + y[1]**2)**1.5 * 0.5 * q * (L*x - x**2) ) / EI
    return np.array([dydx0, dydx1])

# Runge-Kutta 4ª ordem
def rk4_step(x, y, h):
    k1 = f(x, y)
    k2 = f(x + 0.5*h, y + 0.5*h*k1)
    k3 = f(x + 0.5*h, y + 0.5*h*k2)
    k4 = f(x + h, y + h*k3)
    return y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# Resolve PVI com dado y2_0 (theta0)
def solve_ode(theta0):
    x_vals = np.arange(0, L + h, h)
    y_vals = np.zeros((len(x_vals), 2))
    y_vals[0] = np.array([0.0, theta0])
    
    for i in range(len(x_vals) - 1):
        y_vals[i+1] = rk4_step(x_vals[i], y_vals[i], h)
    
    return x_vals, y_vals

# Método do tiro com bissecção
print("="*60)
print("Início do Método do Tiro")
print("="*60)

theta_low = -0.01
theta_high = 0.01

# Primeiras simulações
x_vals, y_vals_low = solve_ode(theta_low)
res_low = y_vals_low[-1, 0]
print(f"\n1ª iteração - theta_low = {theta_low:.6f}, y(L) = {res_low:.6e}")

x_vals, y_vals_high = solve_ode(theta_high)
res_high = y_vals_high[-1, 0]
print(f"1ª iteração - theta_high = {theta_high:.6f}, y(L) = {res_high:.6e}")

# Ajustar limites se necessário
if res_low * res_high > 0:
    print("\n Ajustando limites iniciais: ")
    if abs(res_low) < abs(res_high):
        while res_low * res_high > 0:
            theta_high *= 1.5
            x_vals, y_vals_high = solve_ode(theta_high)
            res_high = y_vals_high[-1, 0]
            print(f"  theta_high = {theta_high:.6f}, y(L) = {res_high:.6e}")
    else:
        while res_low * res_high > 0:
            theta_low *= 1.5
            x_vals, y_vals_low = solve_ode(theta_low)
            res_low = y_vals_low[-1, 0]
            print(f"  theta_low = {theta_low:.6f}, y(L) = {res_low:.6e}")

print("\n" + "="*60)
print("Início da Bissecção")
print("="*60)

# Bissecção
iter_count = 0
while abs(theta_high - theta_low) > tolerance:
    theta_mid = (theta_low + theta_high) / 2.0
    x_vals, y_vals_mid = solve_ode(theta_mid)
    res_mid = y_vals_mid[-1, 0]
    
    print(f"Iteração {iter_count+1}:")
    print(f"  theta_baixo = {theta_low:.8f}, y(L) = {res_low:.3e}")
    print(f"  theta_médio = {theta_mid:.8f}, y(L) = {res_mid:.3e}")
    print(f"  theta_alto = {theta_high:.8f}, y(L) = {res_high:.3e}")
    
    if res_low * res_mid <= 0:
        theta_high = theta_mid
        res_high = res_mid
    else:
        theta_low = theta_mid
        res_low = res_mid
    
    iter_count += 1
    if iter_count > 50:
        print("\nNúmero máximo de iterações atingido")
        break

print("="*60)
print("Resultados finais:")
print("")

print(f"\nConvergência alcançada após {iter_count} iterações")
print(f"Valor ótimo de θ = y'(0) = {theta_mid:.10f} rad")
print(f"Resíduo final y(L) = {res_mid:.10e} m")

# Solução final
x_vals, y_vals_final = solve_ode(theta_mid)

# Calcular deflexão máxima
y_max_idx = np.argmin(y_vals_final[:, 0])  # Índice do valor mais negativo
y_max = y_vals_final[y_max_idx, 0]
x_max = x_vals[y_max_idx]

print(f"\nDeflexão máxima:")
print("")
print(f"  Valor: {y_max:.6f} m")
print(f"  Posição: x = {x_max:.2f} m")

# Calcular solução linearizada para comparação
def solucao_linear(x):
    return -(q/(24*EI)) * (L**3*x - 2*L*x**3 + x**4)

y_linear_max = solucao_linear(L/2)
print("")
print(f"\nComparação com teoria linearizada:")
print(f"  Deflexão máxima linear: {y_linear_max:.6f} m")
print(f"  Diferença percentual: {abs((y_max - y_linear_max)/y_linear_max)*100:.2f}%")

# Valores em pontos específicos
print("")
print("="*60)
print("Deflexão em pontos selecionados:")
print("")

pontos = [0.0, 1.0, 2.0, 3.0, 4.0]
for x_ponto in pontos:
    idx = int(x_ponto / h)
    if idx < len(x_vals):
        y_val = y_vals_final[idx, 0]
        print(f"  x = {x_ponto:.1f} m → y = {y_val:.6f} m")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals_final[:, 0], 'b-', linewidth=2, label='Solução não-linear')

# Adicionar solução linearizada
y_linear = solucao_linear(x_vals)

plt.plot(x_vals, y_linear, 'r--', linewidth=1.5, alpha=0.7, label='Teoria linearizada')

plt.xlabel('x (m)', fontsize=12)
plt.ylabel('Deflexão y(x) (m)', fontsize=12)
plt.title('Deflexão da viga - Comparação entre teorias linear e não-linear', fontsize=14)
plt.grid(True, alpha=0.3)
plt.axhline(0, color='k', linewidth=0.5)
plt.axvline(0, color='k', linewidth=0.5)
plt.axvline(L, color='r', linestyle='--', linewidth=1, label='x = L')
plt.legend(fontsize=11)
plt.tight_layout()

# Salvar figura
plt.savefig('deflexao_viga.png', dpi=300, bbox_inches='tight')
plt.show()
print("="*60)
print("EXECUÇÃO CONCLUÍDA")
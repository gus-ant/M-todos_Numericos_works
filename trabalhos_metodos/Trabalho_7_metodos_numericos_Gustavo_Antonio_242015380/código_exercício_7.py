import numpy as np
import matplotlib.pyplot as plt

m = 2.0       # kg
k = 800.0     # N/m
F0 = 50.0     # N
omega = 3.0   # rad/s

# Condições iniciais
x0 = 0.1      # m
v0 = 0.1      # m/s

# Intervalo de tempo
t0 = 0.0
tf = 10.0
h = 0.01      # passo

# Número de pontos
n = int((tf - t0) / h) + 1

# Vetor tempo
t = np.linspace(t0, tf, n)

# Inicialização dos vetores para cada método
# Euler simples
x_euler = np.zeros(n)
v_euler = np.zeros(n)
x_euler[0] = x0
v_euler[0] = v0

# Euler modificado (RK2)
x_rk2 = np.zeros(n)
v_rk2 = np.zeros(n)
x_rk2[0] = x0
v_rk2[0] = v0

# RK4
x_rk4 = np.zeros(n)
v_rk4 = np.zeros(n)
x_rk4[0] = x0
v_rk4[0] = v0

# Função da EDO: dv/dt = f(t, x, v)
def f(t, x, v):
    return (-k/m)*x + (F0/m)*np.cos(omega*t)

# Método de Euler simples
for i in range(n-1):
    v_euler[i+1] = v_euler[i] + h * f(t[i], x_euler[i], v_euler[i])
    x_euler[i+1] = x_euler[i] + h * v_euler[i]

# Método de Euler modificado (RK2)
for i in range(n-1):
    # Estágio 1
    k1v = f(t[i], x_rk2[i], v_rk2[i])
    k1x = v_rk2[i]
    
    # Estágio 2
    k2v = f(t[i] + h, x_rk2[i] + h*k1x, v_rk2[i] + h*k1v)
    k2x = v_rk2[i] + h*k1v
    
    # Atualização
    v_rk2[i+1] = v_rk2[i] + (h/2)*(k1v + k2v)
    x_rk2[i+1] = x_rk2[i] + (h/2)*(k1x + k2x)

# Método de RK4
for i in range(n-1):
    # Estágio 1
    k1v = f(t[i], x_rk4[i], v_rk4[i])
    k1x = v_rk4[i]
    
    # Estágio 2
    k2v = f(t[i] + h/2, x_rk4[i] + (h/2)*k1x, v_rk4[i] + (h/2)*k1v)
    k2x = v_rk4[i] + (h/2)*k1v
    
    # Estágio 3
    k3v = f(t[i] + h/2, x_rk4[i] + (h/2)*k2x, v_rk4[i] + (h/2)*k2v)
    k3x = v_rk4[i] + (h/2)*k2v
    
    # Estágio 4
    k4v = f(t[i] + h, x_rk4[i] + h*k3x, v_rk4[i] + h*k3v)
    k4x = v_rk4[i] + h*k3v
    
    # Atualização
    v_rk4[i+1] = v_rk4[i] + (h/6)*(k1v + 2*k2v + 2*k3v + k4v)
    x_rk4[i+1] = x_rk4[i] + (h/6)*(k1x + 2*k2x + 2*k3x + k4x)

# Gráficos
plt.figure(figsize=(14, 8))

# Gráfico 1: Posição x(t)
plt.subplot(2, 2, 1)
plt.plot(t, x_euler, 'b-', label='Euler', linewidth=0.5)
plt.plot(t, x_rk2, 'r-', label='RK2', linewidth=0.5)
plt.plot(t, x_rk4, 'g-', label='RK4', linewidth=0.5)
plt.xlabel('Tempo (s)')
plt.ylabel('Posição x(t) (m)')
plt.title('Posição em função do tempo')
plt.legend()
plt.grid(True)

# Gráfico 2: Velocidade v(t)
plt.subplot(2, 2, 2)
plt.plot(t, v_euler, 'b-', label='Euler', linewidth=0.5)
plt.plot(t, v_rk2, 'r-', label='RK2', linewidth=0.5)
plt.plot(t, v_rk4, 'g-', label='RK4', linewidth=0.5)
plt.xlabel('Tempo (s)')
plt.ylabel('Velocidade v(t) (m/s)')
plt.title('Velocidade em função do tempo')
plt.legend()
plt.grid(True)

# Gráfico 3: Comparação final (últimos 2 segundos)
plt.subplot(2, 2, 3)
mask = t >= 8.0  # Últimos 2 segundos
plt.plot(t[mask], x_euler[mask], 'b-', label='Euler', linewidth=1)
plt.plot(t[mask], x_rk2[mask], 'r-', label='RK2', linewidth=1)
plt.plot(t[mask], x_rk4[mask], 'g-', label='RK4', linewidth=1)
plt.xlabel('Tempo (s)')
plt.ylabel('Posição x(t) (m)')
plt.title('Comparação final (t = 8 a 10 s)')
plt.legend()
plt.grid(True)

# Gráfico 4: Erro relativo entre métodos
plt.subplot(2, 2, 4)
erro_rk2_vs_rk4 = np.abs(x_rk2 - x_rk4) / np.maximum(np.abs(x_rk4), 1e-10)
plt.plot(t, erro_rk2_vs_rk4, 'r-', label='Erro RK2 vs RK4', linewidth=0.5)
plt.yscale('log')
plt.xlabel('Tempo (s)')
plt.ylabel('Erro relativo')
plt.title('Erro relativo entre RK2 e RK4')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Exibir últimos valores para comparação
print("Últimos valores (t = 10 s):")
print(f"Euler:   x = {x_euler[-1]:.6f} m, v = {v_euler[-1]:.6f} m/s")
print(f"RK2:     x = {x_rk2[-1]:.6f} m, v = {v_rk2[-1]:.6f} m/s")
print(f"RK4:     x = {x_rk4[-1]:.6f} m, v = {v_rk4[-1]:.6f} m/s")
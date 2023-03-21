# %%
import numpy as np
import matplotlib.pyplot as plt
import math
#import numerical_continuation
# %%
N = 100
x_min = -4
x_max = 4
alpha = 1
beta = 10
h = (x_max-x_min)/N

A = np.zeros((N+1, N+1))
A[0, 0] = 1
A[N, N] = 1
for i in range(1, N):
    A[i, i-1] = 1
    A[i, i] = -2
    A[i, i+1] = 1

b = np.zeros(N+1)
b[0] = alpha
b[1:-1] = 0*h**2
b[-1] = beta

u_pred = np.linalg.solve(A, b)
x_discrete = np.linspace(x_min, x_max, N+1)

u_true = ((beta-alpha)/(x_max-x_min))*(x_discrete-x_min)+alpha

plt.plot(x_discrete, u_pred)
plt.plot(x_discrete, u_true)
# %%
N = 100
x_min = 0
x_max = 1
alpha = 0
beta = 0
h = (x_max-x_min)/N
D = 1
mu = 0.1

x_discrete = np.linspace(x_min, x_max, N+1)
qx = math.e**(mu*x_discrete)

A = np.zeros((N+1, N+1))
A[0, 0] = 1
A[N, N] = 1
for i in range(1, N):
    A[i, i-1] = 1
    A[i, i] = -2
    A[i, i+1] = 1

b = np.zeros(N+1)
b[0] = alpha
b[1:-1] = (-qx[1:-1]/D)*(h**2)
b[-1] = beta

u_pred = np.linalg.solve(A, b)

plt.plot(x_discrete, u_pred)
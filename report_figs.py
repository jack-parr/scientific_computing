# %%
import numpy as np
import matplotlib.pyplot as plt
import math
import ode_solver
import numerical_shooting
import numerical_continuation
import pde_solver
# %%
# 1.1
def func1(x, t, args): # return x_dot
    return np.array([x])

def true1(x):
    return math.e**x

euler_func1 = ode_solver.solve_to(func1, 'euler', [1], 0, 1, 0.05)
rk4_func1 = ode_solver.solve_to(func1, 'rk4', [1], 0, 1, 0.05)
plt.plot(euler_func1[1], euler_func1[0])
plt.plot(rk4_func1[1], rk4_func1[0])
plt.plot(euler_func1[1], true1(euler_func1[1]))
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title('Evaluating ODE $\dot{x} = x$')
plt.legend(['Euler', 'RK4', 'Exact'])
plt.grid()
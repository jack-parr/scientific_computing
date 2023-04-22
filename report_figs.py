# %%
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import timeit
import ode_solver
import numerical_shooting
import numerical_continuation
import pde_solver
# %%
# 1.1 simple
def func1(x, t, args): # return x_dot
    return np.array([x])

def true1(t):
    return math.e**t

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
# %%
# 1.1 system
def func2(x, t, args): # return [x_dot, y_dot]
    return np.array([x[1], -x[0]])

def true2(t):
    x = []
    y = []
    for s in t:
        x.append(math.cos(s) + math.sin(s))
        y.append(math.cos(s) - math.sin(s))
    return np.array([x, y])

euler_func2 = ode_solver.solve_to(func2, 'euler', [1,1], 0, 5, 0.01)
rk4_func2 = ode_solver.solve_to(func2, 'rk4', [1,1], 0, 5, 0.01)
plt.subplot(2,1,1)
plt.plot(euler_func2[-1], euler_func2[0])
plt.plot(rk4_func2[-1], rk4_func2[0])
plt.plot(euler_func2[-1], true2(euler_func2[-1])[0])
plt.ylabel('x(t)')
plt.title('Evaluating ODE $\ddot{x} = x$')
plt.legend(['Euler', 'RK4', 'Exact'])
plt.grid()
plt.subplot(2,1,2)
plt.plot(euler_func2[-1], euler_func2[1])
plt.plot(rk4_func2[-1], rk4_func2[1])
plt.plot(euler_func2[-1], true2(euler_func2[-1])[1])
plt.xlabel('t')
plt.ylabel('y(t)')
plt.yticks([-1, 0, 1])
plt.grid()
# %%
# 1.1 error
t_stor = np.logspace(-5, 0, 100)
euler_err = []
rk4_err = []
for t in t_stor:
    euler_func1 = ode_solver.solve_to(func1, 'euler', [1], 0, 1, t)
    euler_err.append(abs(math.e - euler_func1[0][-1]))
    rk4_func1 = ode_solver.solve_to(func1, 'rk4', [1], 0, 1, t)
    rk4_err.append(abs(math.e - rk4_func1[0][-1]))
errs = np.array([euler_err, rk4_err, t_stor])
# %%
plt.loglog(errs[-1], errs[0], 'o')
plt.loglog(errs[-1], errs[1], 'o')
plt.xlabel('$\Delta t_{max}$')
plt.ylabel('Absolute Error')
plt.title('Error of Solution at $x(1)$')
plt.legend(['Euler', 'RK4'])
idx1 = 21
idx2 = 89
plt.plot(errs[-1][idx1], errs[0][idx1], 'ok', markersize=10)
plt.plot(errs[-1][idx2], errs[1][idx2], 'ok', markersize=10)
plt.plot([errs[-1][0], errs[-1][-1]], [errs[0][idx1], errs[0][idx1]], '--k')
plt.grid()
print(errs[-1][idx1])
print(errs[-1][idx2])
# %%
eulert1 = time.time()
euler_func1 = ode_solver.solve_to(func1, 'euler', [1], 0, 1, 0.000115)
eulert2 = time.time()
rk4t1 = time.time()
rk4_func1 = ode_solver.solve_to(func1, 'rk4', [1], 0, 1, 0.312572)
rk4t2 = time.time()
print(1000*(eulert2-eulert1))
print(1000*(rk4t2-rk4t1))
# %%
# 1.2
def pred_prey(x, t, args):
    a, b, d = args
    x1, x2 = x
    dxdt = np.array([(x1*(1-x1)) - ((a*x1*x2)/(d+x1)), b*x2*(1-(x2/x1))])
    return dxdt

def pp_pc(x, args):
    return pred_prey(x, 0, args)[1]

args=[1, 0.2, 0.1]

pporbit = numerical_shooting.orbit_shoot(
    pred_prey,
    [0.5, 0.5, 20],
    phase_con=pp_pc,
    func_args=args,
    phase_args=args)
print(pporbit)

ppsol = ode_solver.solve_to(pred_prey, 'rk4', [pporbit[0],pporbit[1]], 0, pporbit[2], 0.01, args)
plt.subplot(2,2,1)
plt.plot(ppsol[-1], ppsol[0], ppsol[-1], ppsol[1])
plt.xlabel('t')
plt.title('(a)')
plt.legend(['x', 'y'])
plt.grid()
plt.subplot(2,2,2)
plt.plot(ppsol[0], ppsol[1])
plt.xlabel('x')
plt.ylabel('y')
plt.title('(b)')
plt.tight_layout()
plt.grid()
# %%

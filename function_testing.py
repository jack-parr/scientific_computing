# %%
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import ode_solver as solve_ode
import numerical_shooting as shooting
import pde_solver as solve_pde
import math

#%%
def hopf_func(x, t, args):
    b, s = args
    x1, x2, x3 = x
    dxdt = np.array([((b*x1) - x2 + (s*x1*(x1**2 + x2**2))), (x1 + (b*x2) + (s*x2*(x1**2 + x2**2))), -x3])
    return dxdt

# %%
solve_test = solve_ode.solve_to(hopf_func, 'rk4', [1, 1, -1], 0, 10, 0.1, [1, -1])
print(solve_test[:,0])
# %%
def pred_prey(x, t, args):
    a, b, d = args
    x1, x2 = x
    dxdt = np.array([(x1*(1-x1)) - ((a*x1*x2)/(d+x1)), b*x2*(1-(x2/x1))])
    return dxdt

def pp_pc(x, args):
    return pred_prey(x, 0, args)[1]

# %%
a = 1
b = 0.2
d = 0.1
orbit = shooting.orbit_shoot(pred_prey, [0.6, 0.6, 20], pp_pc, sp.optimize.fsolve, args=[a, b, d])
print(orbit)
# %%
t = 0.2
x = [0.6, 0.6, 20]

# %%
test_solve = solve_ode.solve_to(pred_prey, 'rk4', [1.5, 1], 0, 10, 0.127, args=[1, 0.2, 0.1])
print(len(test_solve))
# %%
# PDE SOLVER
# parameters
D = 1
x_min = 0
x_max = 5
t_min = 0
t_max = 0.5
nx = 100
nt = 1000

def l_bound(x, t, args): # at x=x_min
    return 0

def r_bound(x, t, args): # at x=x_max
    return 0

def initial(x, t, args):
    x_min, x_max = args
    y = np.sin((math.pi * x) / (x_max - x_min))
    return y

output = solve_pde.solve_diffusion('crank_nicolson', 'dirichlet', l_bound, r_bound, initial, D, x_min, x_max, nx, t_min, t_max, nt, init_args=[x_min, x_max])

def heat_exact(x, t, D, x_min, x_max):
    L = x_max - x_min
    return np.exp(-D * t * (math.pi**2 / L**2)) * np.sin((math.pi * (x - x_min)) / (L))

heat_true = heat_exact(output[:,1], t_max, D, x_min, x_max)
plt.plot(output[:,1], heat_true, output[:,1], output[:,0])
# %%
# SOURCE TERM TESTING
D = 1
x_min = 0
x_max = 5
t_min = 0
t_max = 0.5
nx = 100
nt = 1000

def l_bound(x, t, args): # at x=x_min
    return 0

def r_bound(x, t, args): # at x=x_max
    return 0

def initial(x, t, args):
    x_min, x_max = args
    y = np.sin((math.pi * x) / (x_max - x_min))
    return y

output = solve_pde.solve_diffusion('crank_nicolson', 'dirichlet', l_bound, r_bound, initial, D, x_min, x_max, nx, t_min, t_max, nt, init_args=[x_min, x_max])

def heat_exact(x, t, D, x_min, x_max):
    L = x_max - x_min
    return np.exp(-D * t * (math.pi**2 / L**2)) * np.sin((math.pi * (x - x_min)) / (L))

heat_true = heat_exact(output[:,1], t_max, D, x_min, x_max)
plt.plot(output[:,1], heat_true, output[:,1], output[:,0])
# %%

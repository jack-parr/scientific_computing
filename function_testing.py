# %%
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import ode_solver as solve_ode
import numerical_shooting as shooting

#%%
def hopf_func(x, t, args):
    b, s = args
    x1, x2, x3 = x
    dxdt = np.array([((b*x1) - x2 + (s*x1*(x1**2 + x2**2))), (x1 + (b*x2) + (s*x2*(x1**2 + x2**2))), -x3])
    return dxdt

# %%
solve_test = solve_ode.solve_to(hopf_func, 'rk4', [1, 1, -1], 0, 10, 0.1, [1, -1])
print(solve_test)
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
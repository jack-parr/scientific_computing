# %%
import numpy as np
from numerical_shooting import orbit_shoot
import scipy as sp
import matplotlib.pyplot as plt

# %%
# NATURAL PARAMETER CONTINUATION
def func1(x, c):
    return x**3 - x + c

c_all = np.linspace(-2, 2, 100)
r_stor = []

root = sp.optimize.fsolve(func1, x0=5, args=(-2))
r_stor.append(root)
for c in c_all[1:]:
    root = sp.optimize.fsolve(func1, x0=root, args=(c))
    r_stor.append(root)
# %%
plt.plot(c_all, r_stor)

# %%
x = np.linspace(-2, 2, 100)
y = func1(x, 0)
plt.plot(x, y)
plt.grid()
# %%
def continuation(func, x0, par0, vary_par, step_size, max_steps, discretisation, solver):
    r_stor = []
    root = solver(func, x0, args=par0)
    r_stor.append(root)
    for i in range(max_steps):
        vary_par += step_size
        root = solver(func, root, args=(vary_par))
        r_stor.append(root)

    return r_stor

# %%
c = -2

results = continuation(
    func=func1,
    x0=5,
    par0=(c),
    vary_par=c,
    step_size=0.01,
    max_steps=400,
    discretisation=None,
    solver=sp.optimize.fsolve
)
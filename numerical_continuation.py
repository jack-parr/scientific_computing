# %%
import numpy as np
from numerical_shooting import orbit_shoot
from numerical_shooting import shooting_problem
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
def natural_cont(func, x0, args0, vary_par_idx, max_par, num_steps, discretisation, pc, solver):
    r_stor = []
    pars = np.linspace(args0[vary_par_idx], max_par, num_steps)

    for par in pars:
        args0[vary_par_idx] = par
        if pc:
            args_all = (pc, args0)
        else:
            args_all = args0

        root = solver(discretisation(func), x0, args=args_all)
        r_stor.append(root)
        x0 = root

    return np.array([pars, r_stor])


def pseudo_arclength(func, x0, args0, vary_par_idx, max_par, num_steps, discretisation, pc, solver):
    pars = np.linspace(args0[vary_par_idx], max_par, num_steps)

    # find first two values
    x1 = solver(discretisation(func), x0, args=args0)
    args0[vary_par_idx] = pars[1]
    x2 = solver(discretisation(func), x1, args=args0)

    r_stor = [x1, x2]
    par_stor = [pars[0], pars[1]]

    i = 0
    while i < num_steps:
        # get last two values
        x_0, x_1 = r_stor[-2], r_stor[-1]
        p_0, p_1 = par_stor[-2], par_stor[-1]

        # calculate secant
        dx = x_1 - x_0
        dp = p_1 - p_0

        # calulate arclength equation
        x_pred = x_1 + dx
        p_pred = p_1 + dp
        preds = np.array([x_pred, p_pred], dtype='float64')
        

        # update parameter
        args0[vary_par_idx] = p_pred

        # pseudo solve
        def aug_prob(x0, func, pc, discretisation, dx, dp, args0, vary_par_idx):
            x = x0[:-1]
            p = x0[-1]
            args0[vary_par_idx] = p


            g = discretisation(func)(x, args0)[0]
            
            
            arc = np.dot(x_1 - x_pred, dx) + np.dot(p_1 - p_pred, dp)
            f = np.append(g, arc)
            return f
        

        result = solver(aug_prob, preds, args=(func, pc, discretisation, dx, dp, args0, vary_par_idx))
        r_stor.append(result[:-1])
        par_stor.append(result[-1])

        i += 1

    return r_stor, par_stor

# %%
c = -2

test_nat = natural_cont(
    func=func1,
    x0=5,
    args0=[c],
    vary_par_idx=0,
    max_par=2,
    num_steps=400,
    discretisation=(lambda x: x),
    pc=None,
    solver=sp.optimize.fsolve
)
# %%
c = -2

test_pseudo1, test_pseudo2 = pseudo_arclength(
    func=func1,
    x0=5,
    args0=[c],
    vary_par_idx=0,
    max_par=2,
    num_steps=400,
    discretisation=(lambda x: x),
    pc=None,
    solver=sp.optimize.fsolve
)
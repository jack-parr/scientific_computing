# %%
import numpy as np
from numerical_shooting import shooting_problem
import scipy as sp
import matplotlib.pyplot as plt

# %%
# NATURAL PARAMETER CONTINUATION
def func1(x, args):
    c = args[0]
    return x**3 - x + c

# %%

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
    if pc != None:
        a = (pc, args0)
    else:
        a = (args0)
    x1 = solver(discretisation(func), x0, args=a)
    args0[vary_par_idx] = pars[1]
    x2 = solver(discretisation(func), x1, args=a)

    #r_stor = np.empty(shape=(num_steps, len(x1)))*np.nan
    #r_stor[0] = x1
    #r_stor[1] = x2
    r_stor = [x1, x2]
    #par_stor = np.empty(shape=(num_steps, 1))*np.nan
    #par_stor[0] = pars[0]
    #par_stor[1] = pars[1]
    par_stor = [pars[0], pars[1]]

    i = 2
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
        # preds = np.array([x_pred, p_pred], dtype='float64')
        preds = np.append(x_pred, p_pred)
        

        # update parameter
        args0[vary_par_idx] = p_pred

        # pseudo solve
        def aug_prob(x0, func, pc, discretisation, dx, dp, args0, vary_par_idx):
            x = x0[:-1]
            p = x0[-1]
            args0[vary_par_idx] = p

            d = discretisation(func)
            if pc != None:
                g = d(x, pc, args0)
            else:
                g = d(x, args0)
            
            
            arc = np.dot(x - x_pred, dx) + np.dot(p - p_pred, dp)
            f = np.append(g, arc)
            return f
        

        result = solver(aug_prob, preds, args=(func, pc, discretisation, dx, dp, args0, vary_par_idx))
        #r_stor[i] = result[:-1]
        r_stor.append(result[:-1])
        #par_stor[i] = result[-1]
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

plt.plot(test_pseudo2, test_pseudo1)

# %%
def hopf_func(x, t, args):
    b, s = args
    x1, x2 = x
    dxdt = np.array([((b*x1) - x2 + (s*x1*(x1**2 + x2**2))), (x1 + (b*x2) + (s*x2*(x1**2 + x2**2)))])
    return dxdt

def hopf_pc(x, args):
    return hopf_func(x, 0, args)[0]

# %%
hopf_pseudo1, hopf_pseudo2 = pseudo_arclength(
    func=hopf_func,
    x0=[1.4, 1, 6.3],
    args0=[2, -1],
    vary_par_idx=0,
    max_par=-1,
    num_steps=50,
    discretisation=shooting_problem,
    pc=hopf_pc,
    solver=sp.optimize.fsolve
)

plt.plot(hopf_pseudo2, hopf_pseudo1)
# %%
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numerical_shooting import shooting_problem
# %%
def natural_cont(func, x0, init_args, vary_par_idx, max_par, num_steps, discretisation=(lambda x: x), solver=sp.optimize.fsolve):
    """
    Text.
    ----------
    Parameters
    x0 : list
        The guess of the coordinates and time period of a periodic orbit.
    phase_con : function
        Returns the phase condition of the shooting problem.
    func_args : list
        Additional parameters needed by 'func'.
    phase_args : list
        Additional parameters needed by 'phase_con'.
    ----------
    Returns
        Text.
    """

    u_stor = []
    pars = np.linspace(init_args[vary_par_idx], max_par, num_steps)

    for par in pars:
        init_args[vary_par_idx] = par
        root = solver(discretisation(func), x0, args=init_args)
        u_stor.append(root)
        x0 = root

    return np.vstack([np.array(u_stor).T, pars])
# %%
def pseudo_arclength(func, x0, init_args, vary_par_idx, max_par, num_steps, discretisation=(lambda x: x), solver=sp.optimize.fsolve, phase_con=None):

    pars = np.linspace(init_args[vary_par_idx], max_par, num_steps)

    def make_args(phase_con, init_args, vary_par_idx, new_par):
        init_args[vary_par_idx] = new_par
        if phase_con != None:
            a = (phase_con, init_args, init_args)
        else:
            a = (init_args)

        return a
    
    # find first two values
    x1 = solver(discretisation(func), x0, args=make_args(phase_con, init_args, vary_par_idx, pars[0]))
    x2 = solver(discretisation(func), x1, args=make_args(phase_con, init_args, vary_par_idx, pars[1]))

    r_stor = [x1, x2]
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
        preds = np.append(x_pred, p_pred)

        # pseudo solve
        def aug_prob(x0, func, vary_par_idx, dx, dp, discretisation, phase_con, init_args):
            x = x0[:-1]
            p = x0[-1]
            init_args = make_args(phase_con, init_args, vary_par_idx, p)

            d = discretisation(func)
            if phase_con != None:
                g = d(x, init_args[0], init_args[1], init_args[2])
            else:
                g = d(x, init_args)
            
            
            arc = np.dot(x - x_pred, dx) + np.dot(p - p_pred, dp)
            f = np.append(g, arc)
            return f
        

        result = solver(aug_prob, preds, args=(func, vary_par_idx, dx, dp, discretisation, phase_con, init_args))
        r_stor.append(result[:-1])
        par_stor.append(result[-1])

        i += 1

    return np.vstack([np.array(r_stor).T, np.array(par_stor)])

# %%
def func1(x, args):
    c = args[0]
    return x**3 - x + c

c = -2

test1 = pseudo_arclength(
    func=func1,
    x0=5,
    init_args=[c],
    vary_par_idx=0,
    max_par=2,
    num_steps=400,
    discretisation=(lambda x: x),
    solver=sp.optimize.fsolve,
    phase_con=None    
)

plt.plot(test1[-1], test1[0])

# %%
def hopf_func(x, t, args):
    b, s = args
    x1, x2 = x
    dxdt = np.array([((b*x1) - x2 + (s*x1*(x1**2 + x2**2))), (x1 + (b*x2) + (s*x2*(x1**2 + x2**2)))])
    return dxdt

def hopf_pc(x, args):
    return hopf_func(x, 0, args)[0]

test2 = pseudo_arclength(
    func=hopf_func,
    x0=[1.4, 1, 6.3],
    init_args=[2, -1],
    vary_par_idx=0,
    max_par=-1,
    num_steps=50,
    discretisation=shooting_problem,
    solver=sp.optimize.fsolve,
    phase_con=hopf_pc
)

check = test2[0]
plt.plot(test2[-1], test2[0])
# %%

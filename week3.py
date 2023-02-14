# numercial shooting.
# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
import myfunctions as myfunc
# %%
def pred_prey(x, t, args):
    a, b, d = args
    x1, x2 = x
    dxdt = np.array([(x1*(1-x1)) - ((a*x1*x2)/(d+x1)), b*x2*(1-(x2/x1))])
    return dxdt
# %%
a = 1
b = 0.2
d = 0.1
x0 = [0.3, 0.1]
x_sol, t_sol = myfunc.solve_to(pred_prey, 'rk4', x0, 0, 200, 0.1, args=[a, b, d])
x1_sol = [i[0] for i in x_sol]
x2_sol = [i[1] for i in x_sol]
plt.plot(t_sol, x1_sol, t_sol, x2_sol)
plt.legend(["x", "y"])

lc_start = 667
lc_end = lc_start + 199
print('LC START: x = ' + str(x1_sol[lc_start]) + ', y = ' + str(x2_sol[lc_start]))
print('LC END: x = ' + str(x1_sol[lc_end]) + ', y = ' + str(x2_sol[lc_end]))
print('PERIOD = ' + str(t_sol[lc_end] - t_sol[lc_start]))
plt.plot(t_sol[lc_start]*np.ones(2), np.array([0, np.max([np.max(x1_sol), np.max(x2_sol)])]), c='r')
plt.plot(t_sol[lc_end]*np.ones(2), np.array([0, np.max([np.max(x1_sol), np.max(x2_sol)])]), c='r')
# %%
def shooting_prob(func):
    def G(x0, args):
        T = x0[-1]
        x1 = x0[:-1]
        print(x0)
        x_sol, t_sol = myfunc.solve_to(func, 'rk4', x1, 0, T, 0.01, args)
        phase_con = func(x1, 0, args)
        G_vec = [x1[0] - x_sol[-1][0], x1[1] - x_sol[-1][1], phase_con[0]]
        return G_vec
    return G


def orbit_shoot(func, x0, solver, args):
    G = shooting_prob(func)
    orbit = solver(G, x0, args)
    return orbit
# %%
a = 1
b = 0.2
d = 0.1
lc = orbit_shoot(pred_prey, [0.3, 0.25, 35], scipy.optimize.fsolve, args=[a, b, d])
print('---------------')
print(lc)
x_sol, t_sol = myfunc.solve_to(pred_prey, 'rk4', [0.3, 0.25], 0, 100, 0.1, args=[a, b, d])
x1_sol = [i[0] for i in x_sol]
x2_sol = [i[1] for i in x_sol]
plt.plot(x1_sol, x2_sol)
plt.plot(lc[0], lc[1], 'ro')
# %%

# numercial shooting.
# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
import myfunctions as myfunc
# %%
def func1_week_3(x, t, *args):
    a, b, d = args
    x1, x2 = x
    dxdt = np.array([(x1*(1-x1)) - ((a*x1*x2)/(d+x1)), b*x2*(1-(x2/x1))])
    return dxdt
# %%
a = 1
b = 0.2
d = 0.1
t_goal = 200
x0 = [0.3, 0.1]
x_pred = myfunc.solve_to('rk4', func1_week_3, x0, 0, t_goal, 0.1, a, b, d)
x1_pred = [i[0] for i in x_pred]
x2_pred = [i[1] for i in x_pred]
t_all = np.linspace(0,t_goal,(10*t_goal)+1)
plt.plot(t_all, x1_pred, t_all, x2_pred)
plt.legend(["x", "y"])

lc_start = 667
lc_end = lc_start + 199
print('LC START: x = ' + str(x1_pred[lc_start]) + ', y = ' + str(x2_pred[lc_start]))
print('LC END: x = ' + str(x1_pred[lc_end]) + ', y = ' + str(x2_pred[lc_end]))
print('PERIOD = ' + str(t_all[lc_end] - t_all[lc_start]))
plt.plot(t_all[lc_start]*np.ones(2), np.array([0, np.max([np.max(x1_pred), np.max(x2_pred)])]), c='r')
plt.plot(t_all[lc_end]*np.ones(2), np.array([0, np.max([np.max(x1_pred), np.max(x2_pred)])]), c='r')
# %%
# functions needed:
# define problem and boundary conditions
# something that copies fsolve
# phase condition is dx/dt(0) = 0
# %%
def test_func(x, *args):
    x1, x2, x3 = x
    x_pred = myfunc.solve_to('rk4', func1_week_3, [x1, x2], 0, 19.9, 0.1, *args)
    f = np.array([x1 - x_pred[-1][0], x2 - x_pred[-1][1], (x1*(1-x1)) - ((a*x1*x2)/(d+x1))])
    return f
x1_init = 0.6
x2_init = 0.2
root = scipy.optimize.fsolve(test_func, [x1_init, x2_init, (x1_init*(1-x1_init)) - ((a*x1_init*x2_init)/(d+x1_init))], args=(a,b,d))
print(root)
# %%

# numercial shooting.
# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
import myfunctions as myfunc
# %%
a = 1
b = 0.22
d = 0.1
t_goal = 1000
x_pred = myfunc.solve_to('rk4', myfunc.func1_week_3, [0.3, 0.1], 0, t_goal, 0.1, a, b, d)
x1_pred = [i[0] for i in x_pred]
x2_pred = [i[1] for i in x_pred]
t_all = np.linspace(0,t_goal,(10*t_goal)+1)
plt.plot(t_all, x1_pred, t_all, x2_pred)
#plt.xlim(104, 121)
plt.legend(["x", "y"])
# %%
# functions needed:
# define problem and boundary conditions
# something that copies fsolve
# 
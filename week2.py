# initial value solver

#%%
import numpy as np
import matplotlib.pyplot as plt
import math
#%%
def create_func(a, b):
    # a = x_coeff
    return [a, b]


def eval_func(func, x):
    return np.array(math.e**x)


def euler_step(func, x0, t, delta_t):
    # does one euler step
    # x0 = current value, t = current t, = delta_t = timestep.
    grad = x0 + func[1]
    t_new = t + delta_t
    return x0 + delta_t*grad, t_new


def solve_to(func, x1, t1, t_goal, delta_t):
    # solves from x1,t1 to x2,t2
    n = 1
    t = t1
    y = x1
    y_pred = [x1]
    while t < t_goal:
        y, t = euler_step(func, y, t, delta_t)
        y_pred.append(y)
    return y_pred
#%%
func1 = create_func(1, 0)  # x' = x
#%%
x0 = 1
t0 = 0
delta_t = 0.1
t_goal = 4

x = np.linspace(t0, t_goal, 100)
y_true = eval_func(func1, x)

y_pred = solve_to(func1, x0, t0, t_goal, delta_t)
t = np.linspace(t0, t_goal, len(y_pred))
plt.plot(x, y_true, t, y_pred)
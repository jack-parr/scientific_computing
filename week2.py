# initial value solver

# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
# %%
def func1(x, t):
    dxdt = np.array(x)
    return dxdt


def func2(x, t):
    x1, x2 = x
    dxdt = np.array([x2, -x1])
    return dxdt


def euler_step(func, x, t, delta_t):
    # does one euler step
    # x = current value, t = current t, = delta_t = timestep.
    grad = func(x, t)
    t_new = t + delta_t
    return x + grad*delta_t, t_new


def rk4_step(func, x, t, delta_t):
    # does one rk4 step
    # x = current value, t = current t, = delta_t = timestep.
    k1 = grad = func(x, t)
    k2 = grad + (k1*delta_t)/2
    k3 = grad + (k2*delta_t)/2
    k4 = grad + (k3*delta_t)
    t_new = t + delta_t
    return x + (k1/6 + k2/3 + k3/3 + k4/6)*delta_t, t_new


def solve_to(method, func, x0, t0, t_goal, delta_t):
    # solves from x0,t0 to x_goal,t_goal
    t = t0
    x = x0
    x_pred = [x]
    steps = round((t_goal-t0)/delta_t)
    if method == 'euler':    
        for step in range(0, steps):
            x, t = euler_step(func, x, t, delta_t)
            x_pred.append(x)
    elif method == 'rk4':
        for step in range(0, steps):
            x, t = rk4_step(func, x, t, delta_t)
            x_pred.append(x)
    return x_pred
# %%
x0 = 1
t0 = 0
t_goal = 1
n_steps = 10
method = 'euler'

delta_t = (t_goal-t0)/(n_steps)
t = np.linspace(t0, t_goal, n_steps+1)
x_pred = solve_to(method, func1, x0, t0, t_goal, delta_t)

sp_t = np.linspace(t0, t_goal, 101)
sp_odeint = scipy.integrate.odeint(func1, x0, sp_t)
x_true = math.e**sp_t

plt.plot(sp_t, x_true, t, x_pred)
# %%
x0 = 1
t0 = 0
t_goal = 1
x1_true = math.e**t_goal

def t_step_trials(func, x0, t0, t_goal, x1_true):
    delta_t_stor = []
    err_abs_euler_stor = []
    err_abs_rk4_stor = []
    for n_steps in range(1, 100):
        delta_t = (t_goal-t0)/(n_steps)
        delta_t_stor.append(delta_t)

        x_pred_euler = solve_to('euler', func, x0, t0, t_goal, delta_t)
        x1_pred_euler = x_pred_euler[-1]
        err_abs_euler_stor.append(abs(x1_true - x1_pred_euler))

        x_pred_rk4 = solve_to('rk4', func, x0, t0, t_goal, delta_t)
        x1_pred_rk4 = x_pred_rk4[-1]
        err_abs_rk4_stor.append(abs(x1_true - x1_pred_rk4))
    
    plt.loglog(delta_t_stor, err_abs_euler_stor, delta_t_stor, err_abs_rk4_stor)
    plt.xlabel('Time Step (s)')
    plt.ylabel('Absolute Error of x(1) Estimation')
    plt.title('Euler vs RK4')
    plt.legend(['Euler', 'RK4'])

    return delta_t_stor, err_abs_euler_stor, err_abs_rk4_stor
# %%
dt, err_abs_euler, err_abs_rk4 = t_step_trials(func1, x0, t0, t_goal, x1_true)
# %%
x0 = [1, 1]
t0 = 0
t_goal = 1
n_steps = 10
method = 'euler'

delta_t = (t_goal-t0)/(n_steps)
t = np.linspace(t0, t_goal, n_steps+1)
x_pred = solve_to(method, func2, x0, t0, t_goal, delta_t)
plt.plot(t, [i[0] for i in x_pred], t, [i[1] for i in x_pred])

sp_t = np.linspace(t0, t_goal, 101)
sp_odeint = scipy.integrate.odeint(func2, x0, sp_t)
# %%

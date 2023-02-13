import numpy as np
import matplotlib.pyplot as plt
import math


def euler_step(func, x0, t0, delta_t, *args):
    # does one euler step
    # x = current value, t = current t, = delta_t = timestep.
    grad = func(x0, t0, *args)
    t1 = t0 + delta_t
    return x0 + grad*delta_t, t1


def rk4_step(func, x0, t0, delta_t, *args):
    # does one rk4 step
    # x = current value, t = current t, = delta_t = timestep.
    k1 = grad = func(x0, t0, *args)
    k2 = grad + (k1*delta_t)/2
    k3 = grad + (k2*delta_t)/2
    k4 = grad + (k3*delta_t)
    t1 = t0 + delta_t
    return x0 + (k1/6 + k2/3 + k3/3 + k4/6)*delta_t, t1


def solve_to(func, method, x1, t1, t2, deltat_max, *args):
    # solves from x1,t1 to x2,t2.
    if method == 'euler':
        fstep = euler_step
    elif method == 'rk4':
        fstep = rk4_step

    x_sol = np.empty(shape=(math.ceil((t2-t1)/deltat_max)+2, len(x1)))
    t_sol = np.empty(shape=(math.ceil((t2-t1)/deltat_max)+2, len(x1)))
    x_sol[0, :] = x1
    t_sol[0, :] = t1

    i = 1
    while t2 - t1 > deltat_max:
        x1, t1 = fstep(func, x1, t1, deltat_max, *args)
        x_sol[i, :] = x1
        t_sol[i, :] = t1
        i += 1
    else:
        x1, t1 = fstep(func, x1, t1, t2-t1, *args)
        x_sol[i, :] = x1
        t_sol[i, :] = t1
        i += 1

    return x_sol, t_sol


def t_step_trials(func, x0, t0, t_goal, x1_true, *args):
    delta_t_stor = []
    err_abs_euler_stor = []
    err_abs_rk4_stor = []
    for n_steps in range(1, 100):
        delta_t = (t_goal-t0)/(n_steps)
        delta_t_stor.append(delta_t)

        x_pred_euler = solve_to('euler', func, x0, t0, t_goal, delta_t, *args)
        x1_pred_euler = x_pred_euler[-1]
        err_abs_euler_stor.append(abs(x1_true - x1_pred_euler))

        x_pred_rk4 = solve_to('rk4', func, x0, t0, t_goal, delta_t, *args)
        x1_pred_rk4 = x_pred_rk4[-1]
        err_abs_rk4_stor.append(abs(x1_true - x1_pred_rk4))
    
    plt.loglog(delta_t_stor, err_abs_euler_stor, delta_t_stor, err_abs_rk4_stor)
    plt.xlabel('Time Step (s)')
    plt.ylabel('Absolute Error of x(1) Estimation')
    plt.title('Euler vs RK4')
    plt.legend(['Euler', 'RK4'])

    return delta_t_stor, err_abs_euler_stor, err_abs_rk4_stor
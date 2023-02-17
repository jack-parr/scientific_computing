import numpy as np
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


def solve_to(func, method, x1, t1, t2, deltat_max, args):
    # solves from x1,t1 to x2,t2.
    if method == 'euler':
        fstep = euler_step
    elif method == 'rk4':
        fstep = rk4_step

    x_sol = np.empty(shape=(math.floor((t2-t1)/deltat_max)+2, len(x1)+1))*np.nan
    x_sol[0][:len(x1)] = x1
    x_sol[0][-1] = t1

    i = 1
    while not math.isclose(t1, t2):
        x1, t1 = fstep(func, x1, t1, min([t2-t1, deltat_max]), args)
        x_sol[i][:len(x1)] = x1
        x_sol[i][-1] = t1
        i += 1

    return x_sol[~np.isnan(x_sol).any(axis=1), :]


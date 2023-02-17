import numpy as np
import matplotlib.pyplot as plt
import ode_solver as solve_ode


def shooting_problem(func):
    def G(x0, phase_con, args):
        T = x0[-1]
        x1 = x0[:-1]
        x_sol = solve_ode.solve_to(func, 'rk4', x1, 0, T, 0.01, args)
        return np.append(x1 - x_sol[-1][:-1], phase_con(x1, args))
    return G


def orbit_shoot(func, x0, phase_con, solver, args):
    G = shooting_problem(func)
    orbit = solver(G, x0, (phase_con, args))
    return orbit
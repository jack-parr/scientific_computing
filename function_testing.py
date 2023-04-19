# %%
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math
import ode_solver as solve_ode
import numerical_shooting as shooting
import numerical_continuation as num_con
import pde_solver as solve_pde

# %%
# NATURAL PARAMETER CONTINUATION
def func1(x, args):
    c = args[0]
    return x**3 - x + c
# %%
# TRUTH PLOTS
c_all = np.linspace(-2, 2, 100)
r_stor = []

root = sp.optimize.fsolve(func1, x0=5, args=[-2])
r_stor.append(root)
for c in c_all[1:]:
    root = sp.optimize.fsolve(func1, x0=root, args=[c])
    r_stor.append(root)

plt.plot(c_all, r_stor)
# %%
x = np.linspace(-2, 2, 100)
y = func1(x, [0])
plt.plot(x, y)
plt.grid()
# %%
# NO DISCRETISATION
c = -2

test_nat = num_con.natural_cont(
    func=func1,
    x0=5,
    init_args=[c],
    vary_par_idx=0,
    max_par=2,
    num_steps=400,
    solver=sp.optimize.fsolve
)

plt.plot(test_nat[-1], test_nat[0])
# %%
# PSEUDO
# NO DISCRETISATION
def func1(x, args):
    c = args[0]
    return x**3 - x + c

c = -2

test1 = num_con.pseudo_arclength(
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
# WITH PHASE CONDITION AND DISCRETISATION
def hopf_func(x, t, args):
    b, s = args
    x1, x2 = x
    dxdt = np.array([((b*x1) - x2 + (s*x1*(x1**2 + x2**2))), (x1 + (b*x2) + (s*x2*(x1**2 + x2**2)))])
    return dxdt

def hopf_pc(x, args):
    return hopf_func(x, 0, args)[0]

test2 = num_con.pseudo_arclength(
    func=hopf_func,
    x0=[1.4, 1, 6.3],
    init_args=[2, -1],
    vary_par_idx=0,
    max_par=-1,
    num_steps=50,
    discretisation=shooting.shooting_problem,
    solver=sp.optimize.fsolve,
    phase_con=hopf_pc
)

check = test2[0]
plt.plot(test2[-1], test2[0])
# %%
def hopf_func(x, t, args):
    b, s = args
    x1, x2, x3 = x
    dxdt = np.array([((b*x1) - x2 + (s*x1*(x1**2 + x2**2))), (x1 + (b*x2) + (s*x2*(x1**2 + x2**2))), -x3])
    return dxdt

solve_test = solve_ode.solve_to(hopf_func, 'rk4', np.array([1, 1, -1]), 0, 10, 0.1, [1, -1])
plt.plot(solve_test[-1], solve_test[0])
# %%
def pred_prey(x, t, args):
    a, b, d = args
    x1, x2 = x
    dxdt = np.array([(x1*(1-x1)) - ((a*x1*x2)/(d+x1)), b*x2*(1-(x2/x1))])
    return dxdt

def pp_pc(x, args):
    return pred_prey(x, 0, args)[1]

# %%
a = 1
b = 0.2
d = 0.1
orbit = shooting.orbit_shoot(pred_prey, [0.6, 0.6, 20], sp.optimize.fsolve, pp_pc, func_args=[a, b, d], phase_args=[a, b, d])
print(orbit)
# %%
t = 0.2
x = [0.6, 0.6, 20]

# %%
test_solve = solve_ode.solve_to(pred_prey, 'rk4', [1.5, 1], 0, 10, 0.127, args=[1, 0.2, 0.1])
plt.plot(test_solve[2], test_solve[0])
# %%
# PDE SOLVER
# parameters
D = 1
x_min = 0
x_max = 5
t_min = 0
t_max = 0.5
nx = 100
nt = 1000

def l_bound(x, t, args): # at x=x_min
    return 0

def r_bound(x, t, args): # at x=x_max
    return 0

def initial(x, t, args):
    x_min, x_max = args
    y = np.sin((math.pi * x) / (x_max - x_min))
    return y

output = solve_pde.solve_diffusion('crank_nicolson', 'dirichlet', l_bound, r_bound, initial, D, x_min, x_max, nx, t_min, t_max, nt, init_args=[x_min, x_max])

def heat_exact(x, t, D, x_min, x_max):
    L = x_max - x_min
    return np.exp(-D * t * (math.pi**2 / L**2)) * np.sin((math.pi * (x - x_min)) / (L))

heat_true = heat_exact(output[1], t_max, D, x_min, x_max)
#plt.plot(output[:,1], heat_true, output[:,1], output[:,0])
plt.plot(output[1], heat_true, output[1], output[0])
# %%
# SOURCE TERM TESTING
D = 1
x_min = 0
x_max = 1
t_min = 0
t_max = 0.2
nx = 100
nt = 1000
mu = 2

def l_bound(x, t, args): # at x=x_min
    return 0

def r_bound(x, t, args): # at x=x_max
    return 0

def initial(x_arr, t, args):
    return np.zeros(len(x_arr))

def source(x_arr, t, u, args):
    mu = args
    return np.ones(shape=len(x_arr)) * (math.e**(u * mu))

output = solve_pde.solve_diffusion('crank_nicolson', 'dirichlet', l_bound, r_bound, initial, D, x_min, x_max, nx, t_min, t_max, nt, source_func=source, source_args=[mu])

#def bratu_exact(x, t, D, x_min, x_max):
#    L = x_max - x_min
#    return np.exp(-D * t * (math.pi**2 / L**2)) * np.sin((math.pi * (x - x_min)) / (L))

#bratu_true = bratu_exact(output[:,1], t_max, D, x_min, x_max)
#plt.plot(output[:,1], bratu_true, output[:,1], output[:,0])

plt.plot(output[:,1], output[:,0])
# %%
x_arr = np.linspace(x_min, x_max, nx+1)
test2 = source(x_arr[1:nx], 0, [0, 2])

dt = (t_max - t_min) / nt
size = nx-1
def make_b(t):
    b = np.zeros(size)
    b[0] = 0
    b[-1] = 0
    return b + dt*source(x_arr[1:nx], t, (0, 2))
    #return b

testb = make_b(0)
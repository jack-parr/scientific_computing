# %%
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math
import time
import ode_solver
import numerical_shooting
import numerical_continuation
import pde_solver
# %%
# 1.1 simple
def func1(x, t, args): # return x_dot
    return np.array([x])

def true1(t):
    return math.e**t

euler_func1 = ode_solver.solve_to(func1, 'euler', [1], 0, 1, 0.05)
rk4_func1 = ode_solver.solve_to(func1, 'rk4', [1], 0, 1, 0.05)
plt.plot(euler_func1[1], euler_func1[0])
plt.plot(rk4_func1[1], rk4_func1[0])
plt.plot(euler_func1[1], true1(euler_func1[1]))
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title('Evaluating ODE $\dot{x} = x$')
plt.legend(['Euler', 'RK4', 'Exact'])
plt.grid()
# %%
# 1.1 system
def func2(x, t, args): # return [x_dot, y_dot]
    return np.array([x[1], -x[0]])

def true2(t):
    x = []
    y = []
    for s in t:
        x.append(math.cos(s) + math.sin(s))
        y.append(math.cos(s) - math.sin(s))
    return np.array([x, y])

euler_func2 = ode_solver.solve_to(func2, 'euler', [1,1], 0, 5, 0.01)
rk4_func2 = ode_solver.solve_to(func2, 'rk4', [1,1], 0, 5, 0.01)
plt.subplot(2,1,1)
plt.plot(euler_func2[-1], euler_func2[0])
plt.plot(rk4_func2[-1], rk4_func2[0])
plt.plot(euler_func2[-1], true2(euler_func2[-1])[0])
plt.ylabel('x(t)')
plt.title('Evaluating ODE $\ddot{x} = x$')
plt.legend(['Euler', 'RK4', 'Exact'])
plt.grid()
plt.subplot(2,1,2)
plt.plot(euler_func2[-1], euler_func2[1])
plt.plot(rk4_func2[-1], rk4_func2[1])
plt.plot(euler_func2[-1], true2(euler_func2[-1])[1])
plt.xlabel('t')
plt.ylabel('y(t)')
plt.yticks([-1, 0, 1])
plt.grid()
# %%
# 1.1 error
t_stor = np.logspace(-5, 0, 100)
euler_err = []
rk4_err = []
for t in t_stor:
    euler_func1 = ode_solver.solve_to(func1, 'euler', [1], 0, 1, t)
    euler_err.append(abs(math.e - euler_func1[0][-1]))
    rk4_func1 = ode_solver.solve_to(func1, 'rk4', [1], 0, 1, t)
    rk4_err.append(abs(math.e - rk4_func1[0][-1]))
errs = np.array([euler_err, rk4_err, t_stor])
# %%
plt.loglog(errs[-1], errs[0], 'o')
plt.loglog(errs[-1], errs[1], 'o')
plt.xlabel('$\Delta t_{max}$')
plt.ylabel('Absolute Error')
plt.title('Error of Solution at $x(1)$')
plt.legend(['Euler', 'RK4'])
idx1 = 21
idx2 = 89
plt.plot(errs[-1][idx1], errs[0][idx1], 'ok', markersize=10)
plt.plot(errs[-1][idx2], errs[1][idx2], 'ok', markersize=10)
plt.plot([errs[-1][0], errs[-1][-1]], [errs[0][idx1], errs[0][idx1]], '--k')
plt.grid()
print(errs[-1][idx1])
print(errs[-1][idx2])
# %%
eulert1 = time.time()
euler_func1 = ode_solver.solve_to(func1, 'euler', [1], 0, 1, 0.000115)
eulert2 = time.time()
rk4t1 = time.time()
rk4_func1 = ode_solver.solve_to(func1, 'rk4', [1], 0, 1, 0.312572)
rk4t2 = time.time()
print(1000*(eulert2-eulert1))
print(1000*(rk4t2-rk4t1))
# %%
# 1.2
def pred_prey(x, t, args):
    a, b, d = args
    x1, x2 = x
    dxdt = np.array([(x1*(1-x1)) - ((a*x1*x2)/(d+x1)), b*x2*(1-(x2/x1))])
    return dxdt

def pp_pc(x, args):
    return pred_prey(x, 0, args)[1]

args=[1, 0.2, 0.1]

pporbit = numerical_shooting.orbit_shoot(
    pred_prey,
    [0.5, 0.5, 20],
    phase_con=pp_pc,
    func_args=args,
    phase_args=args)
print(pporbit)

ppsol = ode_solver.solve_to(pred_prey, 'rk4', [pporbit[0],pporbit[1]], 0, pporbit[2], 0.01, args)
plt.subplot(2,2,1)
plt.plot(ppsol[-1], ppsol[0], ppsol[-1], ppsol[1])
plt.xlabel('t')
plt.title('(a)')
plt.legend(['x', 'y'])
plt.grid()
plt.subplot(2,2,2)
plt.plot(ppsol[0], ppsol[1])
plt.xlabel('x')
plt.ylabel('y')
plt.title('(b)')
plt.tight_layout()
plt.grid()
# %%
# 1.3 cubic
def cubic(x, args):
    c = args[0]
    return x**3 - x + c

cubicnat = numerical_continuation.natural_continuation(
    cubic,
    [5],
    [-2],
    0,
    2,
    100
)

cubicpseudo = numerical_continuation.pseudo_arclength(
    cubic,
    [5],
    [-2],
    0,
    2,
    100
)

plt.plot(cubicnat[-1], cubicnat[0])
plt.plot(cubicpseudo[-1], cubicpseudo[0])
plt.xlabel('c')
plt.ylabel('Root')
plt.title('Numerical Continuation on the Algebraic Cubic Equation')
plt.legend(['Natural Parameter', 'Pseudo-arclength'])
plt.grid()
# %%
# 1.3 hopf
def hopf_normal(x, t, args):
    b = args[0]
    x1, x2 = x
    return np.array([b*x1 - x2 - x1*(x1**2 + x2**2), x1 + b*x2 - x2*(x1**2 + x2**2)])

def hopf_pc(x, args):
    return hopf_normal(x, 0, args)[0]

hopfnat = numerical_continuation.natural_continuation(
    hopf_normal,
    [1.4, 1, 6.3],
    [2],
    0,
    -1,
    50,
    numerical_shooting.shooting_problem,
    phase_con=hopf_pc
)

hopfpseudo = numerical_continuation.pseudo_arclength(
    func=hopf_normal,
    x0=[1.4, 1, 6.3],
    init_args=[2],
    vary_par_idx=0,
    max_par=-1,
    num_steps=50,
    discretisation=numerical_shooting.shooting_problem,
    phase_con=hopf_pc
)
# %%
plt.plot(hopfnat[-1], hopfnat[0])
plt.plot(hopfpseudo[-1], hopfpseudo[0])
plt.xlabel(r'$\beta$')
plt.ylabel('$\dot{x}$')
plt.title('Numerical Continuation on the Hopf Normal Equations')
plt.legend(['Natural Parameter', 'Pseudo-arclength'])
plt.grid()
# %%
# NOT USED bratu
def bratu_l_bound(x, args):
    return 0

def bratu_r_bound(x, args):
    return 0

def bratu_init(x, args):
    D, a, b, gamma1, gamma2 = args
    return (-1/(2*D)) * (x-a) * (x-b) + ((gamma2-gamma1)/(b-a)) * (x-a) + gamma1

def bratu_source(x, u, args):
    mu = args[0]
    return math.e**(mu * u)

bratu_sp = method_of_lines.solve_bvp(
    method='scipy', 
    boundary_type='dirichlet', 
    l_bound_func=bratu_l_bound, 
    r_bound_func=bratu_r_bound, 
    init_func=bratu_init, 
    D=1, 
    x_min=0, 
    x_max=1, 
    nx=100, 
    source_func=bratu_source,
    init_args=[1, 0, 1, 0, 0],
    source_args=[0.1]
    )

bratu_euler = method_of_lines.solve_bvp(
    method='euler', 
    boundary_type='dirichlet', 
    l_bound_func=bratu_l_bound, 
    r_bound_func=bratu_r_bound, 
    init_func=bratu_init, 
    D=1, 
    x_min=0, 
    x_max=1, 
    nx=100, 
    source_func=bratu_source,
    init_args=[1, 0, 1, 0, 0],
    source_args=[0.1]
    )

bratu_rk4 = method_of_lines.solve_bvp(
    method='rk4', 
    boundary_type='dirichlet', 
    l_bound_func=bratu_l_bound, 
    r_bound_func=bratu_r_bound, 
    init_func=bratu_init, 
    D=1, 
    x_min=0, 
    x_max=1, 
    nx=100, 
    source_func=bratu_source,
    init_args=[1, 0, 1, 0, 0],
    source_args=[0.1]
    )

def bratu_true(x, D, a, b, gamma1, gamma2):
    return (-1/(2*D)) * (x-a) * (x-b) + ((gamma2-gamma1)/(b-a)) * (x-a) + gamma1

plt.plot(bratu_sp[-1], bratu_sp[0])
plt.plot(bratu_euler[-1], bratu_euler[0])
plt.plot(bratu_rk4[-1], bratu_rk4[0])
plt.plot(bratu_sp[-1], bratu_true(bratu_sp[-1], 1, 0, 1, 0, 0))
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Solution of Bratu with $\mu$ = 0.1')
plt.legend(['scipy.optimize.root', 'euler', 'rk4', 'exact'])
plt.grid()
# %%
# NOT USED bratu with num con
def bratu_con_problem(x, args):
    mu = args[0]

    x_pred = method_of_lines.solve_bvp(
        method='scipy', 
        boundary_type='dirichlet', 
        l_bound_func=bratu_l_bound, 
        r_bound_func=bratu_r_bound, 
        init_func=bratu_init, 
        D=1, 
        x_min=0, 
        x_max=1, 
        nx=100, 
        source_func=bratu_source,
        init_args=[1, 0, 1, 0, 0],
        source_args=[mu]
        )

    return x-max(x_pred[0])

bratu_con = numerical_continuation.pseudo_arclength(
    func=bratu_con_problem,
    x0=[0.1],
    init_args=[0],
    vary_par_idx=0,
    max_par=4,
    num_steps=100
)

plt.plot(bratu_con[-1], bratu_con[0])
plt.xlabel('$\mu$')
plt.ylabel('max value of u(x)')
plt.title('Pseudo-arclength on problems using BVP solver.')
plt.grid()
# %%
# 1.4 diffusion linear
def diff_l_bound(x, t, args):
    return 0

def diff_r_bound_dirneu(x, t, args):
    return 0

def diff_r_bound_rob(x, t, args):
    delta, gamma = args
    return delta - (gamma*x)

def diff_init(x, t, args):
    y = np.sin(np.pi * x)
    return y
# %%
xmol = pde_solver.solve_diffusion(
            method='lines', 
            boundary_type='dirichlet', 
            l_bound_func=diff_l_bound, 
            r_bound_func=diff_r_bound_dirneu, 
            init_func=diff_init, 
            D=0.1, 
            x_min=0, 
            x_max=1, 
            nx=100, 
            t_min=0, 
            t_max=2, 
            nt=200, 
            use_sparse=False
            )

xee = pde_solver.solve_diffusion(
            method='explicit_euler', 
            boundary_type='dirichlet', 
            l_bound_func=diff_l_bound, 
            r_bound_func=diff_r_bound_dirneu, 
            init_func=diff_init, 
            D=0.1, 
            x_min=0, 
            x_max=1, 
            nx=100, 
            t_min=0, 
            t_max=2, 
            nt=200, 
            use_sparse=False
            )

xie = pde_solver.solve_diffusion(
            method='implicit_euler', 
            boundary_type='dirichlet', 
            l_bound_func=diff_l_bound, 
            r_bound_func=diff_r_bound_dirneu, 
            init_func=diff_init, 
            D=0.1, 
            x_min=0, 
            x_max=1, 
            nx=100, 
            t_min=0, 
            t_max=2, 
            nt=200, 
            use_sparse=False
            )

xcn = pde_solver.solve_diffusion(
            method='crank_nicolson', 
            boundary_type='dirichlet', 
            l_bound_func=diff_l_bound, 
            r_bound_func=diff_r_bound_dirneu, 
            init_func=diff_init, 
            D=0.1, 
            x_min=0, 
            x_max=1, 
            nx=100, 
            t_min=0, 
            t_max=2, 
            nt=200, 
            use_sparse=False
            )

plt.plot(xmol[-1], xmol[0])
plt.plot(xee[-1], xee[0])
plt.plot(xie[-1], xie[0])
plt.plot(xcn[-1], xcn[0])
plt.plot(0.5, math.e**(-0.2 * math.pi * math.pi), 'ok')
plt.xlabel('x')
plt.ylabel('u(x,2)')
plt.title('Solving the Linear Diffusion PDE')
plt.legend(['Method of Lines', 'Explicit Euler', 'Implicit Euler', 'Crank-Nicolson', 'u(0.5,2) Exact'])
plt.grid()
# %%
methods = ['lines', 'explicit_euler', 'implicit_euler', 'crank_nicolson']
sparses = [False, True]

t_stor = []
for m in methods:
    m_stor = []
    for s in sparses:
        t1 = time.time()
        xmol = pde_solver.solve_diffusion(
            method=m, 
            boundary_type='dirichlet', 
            l_bound_func=diff_l_bound, 
            r_bound_func=diff_r_bound_dirneu, 
            init_func=diff_init, 
            D=0.1, 
            x_min=0, 
            x_max=1, 
            nx=100, 
            t_min=0, 
            t_max=2, 
            nt=200, 
            use_sparse=s
            )
        t2 = time.time()
        print(s)
        print(t2-t1)
        m_stor.append(t2-t1)
    t_stor.append(m_stor)
    print('-----------')

ts = np.array(t_stor)
print(t_stor)
# %%
# PDE WITH NUM CON
def bratu_l_bound(x, args):
    return 0

def bratu_r_bound(x, args):
    return 0

def bratu_init(x, args):
    D, a, b, gamma1, gamma2 = args
    return (-1/(2*D)) * (x-a) * (x-b) + ((gamma2-gamma1)/(b-a)) * (x-a) + gamma1

def bratu_source(x, args):
    mu = args[0]
    return math.e**(mu * x)
# %%
def testsourcefunc(x, args):
    mu = args[0]
    return np.e**(x * mu)

def findiff(boundary_type, D, x_min, x_max, nx, l_bound, r_bound, source, source_args):

    if boundary_type == 'dirichlet':
        size = nx-1
    
    dx = (x_max - x_min) / nx
    x_arr = np.linspace(x_min, x_max, nx+1)

    A = pde_solver.sparse_A(size, -2*D, D)
    b = np.zeros(size)
    b[0] = l_bound
    b[-1] = r_bound
    b = b*D
    q = source(x_arr[1:-1], source_args)

    sol = sp.sparse.linalg.spsolve(A, -b - (dx**2)*q)
    sol = np.concatenate((np.array([l_bound]), sol, np.array([r_bound])))

    return np.vstack([sol, x_arr])

D = 1
x_min = 0
x_max = 1
nx = 100
l = 0
r = 0
source_args = [0]

test = findiff('dirichlet', D, x_min, x_max, nx, l, r, testsourcefunc, source_args)

def truth(x, D, a, b, l, r):
    #return ((r - l)/(b - a))*(x - a) + l
    return (-1/(2*D)) * (x-a) * (x-b) + ((r-l)/(b-a)) * (x-a) + l

plt.plot(test[-1], test[0])
plt.plot(test[-1], truth(test[-1], D, x_min, x_max, l, r))

# %%
def bratu_pseudo(x, args):
    mu = args[0]
    sol = pde_solver.solve_diffusion('dirichlet', 1, 0, 1, 100, 0, 0, bratu_source, [mu])
    return np.array([x-max(sol[0])])

test = numerical_continuation.pseudo_arclength(
    func=bratu_pseudo,
    x0=[1],
    init_args=[0],
    vary_par_idx=0,
    max_par=4,
    num_steps=50,
)
plt.plot(test[-1], test[0])
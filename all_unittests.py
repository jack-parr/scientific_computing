import unittest
import numpy as np
import scipy as sp
import math
import ode_solver
import numerical_shooting
import numerical_continuation
import pde_solver


# DEFINE FUNCTIONS
def pred_prey(x, t, args):
    a, b, d = args
    x1, x2 = x
    dxdt = np.array([(x1*(1-x1)) - ((a*x1*x2)/(d+x1)), b*x2*(1-(x2/x1))])
    return dxdt


def pred_prey_phase_con(x, args):
    return pred_prey(x, 0, args)[1]


def algebraic_cubic(x, args):
    c = args[0]
    return x**3 - x + c


def hopf_normal(x, t, args):
    b = args[0]
    x1, x2 = x
    return np.array([b*x1 - x2 - x1*(x1**2 + x2**2), x1 + b*x2 - x2*(x1**2 + x2**2)])


def hopf_pc(x, args):
    return hopf_normal(x, 0, args)[0]


def diff_l_bound(x, t, args):
    return 0


def diff_r_bound(x, t, args):
    delta, gamma = args
    return delta - (gamma*x)


def diff_init(x, t, args):
    x_min, x_max = args
    y = np.sin((math.pi * x) / (x_max - x_min))
    return y


def diff_source(x_arr, t, u, args):
    mu = args
    return np.ones(shape=len(x_arr)) * (math.e**(u * mu))


# DEFINE TEST METHODS
class test_methods_solve_ode(unittest.TestCase):
    
    def test_euler_step(self):

        x_pred, t = ode_solver.euler_step(
            func=pred_prey, 
            x0=[1.5, 1], 
            t0=0, 
            delta_t=1, 
            args=[1, 0.2, 0.1]
        )
        self.assertEqual(len(x_pred), 2)
        self.assertEqual(round(x_pred[0], 3), -0.188)
        self.assertEqual(round(x_pred[1], 3), 1.067)
        self.assertEqual(int(t), 1)
    

    def test_rk4_step(self):

        x_pred, t = ode_solver.rk4_step(
            func=pred_prey, 
            x0=[1.5, 1], 
            t0=0, 
            delta_t=1, 
            args=[1, 0.2, 0.1]
        )
        self.assertEqual(len(x_pred), 2)
        self.assertEqual(round(x_pred[0], 3), -1.383)
        self.assertEqual(round(x_pred[1], 3), 1.114)
        self.assertEqual(int(t), 1)
    

    def test_solve_to(self):
        
        x_pred = ode_solver.solve_to(
            func=pred_prey, 
            method='euler', 
            x0=[1.5, 1], 
            t0=0, 
            t1=10, 
            deltat_max=0.127, 
            args=[1, 0.2, 0.1]
        )
        self.assertEqual(np.shape(x_pred), (3,80))
        self.assertTrue(np.isclose(x_pred[-1][-1], 10))

        x_pred = ode_solver.solve_to(
            func=pred_prey, 
            method='euler', 
            x0=[1.5, 1], 
            t0=0, 
            t1=10, 
            deltat_max=0.01, 
            args=[1, 0.2, 0.1]
        )
        self.assertEqual(np.shape(x_pred), (3,1001))
        self.assertTrue(np.isclose(x_pred[-1][-1], 10))
        self.assertAlmostEqual(round(x_pred[0][-1], 3), 0.508)
        self.assertAlmostEqual(round(x_pred[1][-1], 3), 0.102)

        x_pred = ode_solver.solve_to(
            func=pred_prey, 
            method='rk4', 
            x0=[1.5, 1], 
            t0=0, 
            t1=10, 
            deltat_max=0.127, 
            args=[1, 0.2, 0.1]
        )
        self.assertEqual(np.shape(x_pred), (3,80))
        self.assertTrue(np.isclose(x_pred[-1][-1], 10))

        x_pred = ode_solver.solve_to(
            func=pred_prey, 
            method='rk4', 
            x0=[1.5, 1], 
            t0=0, 
            t1=10,
            deltat_max=0.01, 
            args=[1, 0.2, 0.1]
        )
        self.assertEqual(np.shape(x_pred), (3,1001))
        self.assertTrue(np.isclose(x_pred[-1][-1], 10))
        self.assertAlmostEqual(round(x_pred[0][-1], 3), 0.516)
        self.assertAlmostEqual(round(x_pred[1][-1], 3), 0.103)


class test_methods_numerical_shooting(unittest.TestCase):

    def test_orbit_shoot(self):
        x0 = [0.6, 0.6, 20]
        x_pred = numerical_shooting.orbit_shoot(
            pred_prey, 
            x0, 
            sp.optimize.fsolve, 
            pred_prey_phase_con, 
            func_args=[1, 0.2, 0.1], 
            phase_args=[1, 0.2, 0.1]
        )
        self.assertEqual(len(x_pred), 3)
        self.assertAlmostEqual(round(x_pred[0], 3), 0.383)
        self.assertAlmostEqual(round(x_pred[1], 3), 0.383)
        self.assertAlmostEqual(round(x_pred[2], 3), 20.724)


class test_methods_numerical_continuation(unittest.TestCase):
 
    def test_natural_continuation(self):

        x_pred = numerical_continuation.natural_continuation(
            func=algebraic_cubic,
            x0=[5],
            init_args=[-2],
            vary_par_idx=0,
            max_par=2,
            num_steps=400,
            solver=sp.optimize.fsolve
        )
        self.assertEqual(np.shape(x_pred), (2, 400))
        self.assertAlmostEqual(round(x_pred[0][-1], 3), 0.577)
        self.assertAlmostEqual(round(x_pred[1][-1], 3), 2.000)

        x_pred = numerical_continuation.natural_continuation(
            func=hopf_normal,
            x0=[1.4, 1, 6.3],
            init_args=[2],
            vary_par_idx=0,
            max_par=-1,
            num_steps=50,
            discretisation=numerical_shooting.shooting_problem,
            solver=sp.optimize.fsolve,
            phase_con=hopf_pc
        )
        self.assertEqual(np.shape(x_pred), (4, 50))
        self.assertAlmostEqual(round(x_pred[0][-1], 3), 0.000)
        self.assertAlmostEqual(round(x_pred[1][-1], 3), 0.000)
        self.assertAlmostEqual(round(x_pred[2][-1], 3), 6.252)
        self.assertAlmostEqual(round(x_pred[3][-1], 3), -1.000)
    

    def test_pseudo_arclength(self):

        x_pred = numerical_continuation.pseudo_arclength(
            func=algebraic_cubic,
            x0=[5],
            init_args=[-2],
            vary_par_idx=0,
            max_par=2,
            num_steps=400,
            discretisation=(lambda x: x),
            solver=sp.optimize.fsolve,
            phase_con=None    
        )
        self.assertEqual(np.shape(x_pred), (2, 400))
        self.assertAlmostEqual(round(x_pred[0][-1], 3), -0.573)
        self.assertAlmostEqual(round(x_pred[1][-1], 3), -0.385)

        x_pred = numerical_continuation.pseudo_arclength(
            func=hopf_normal,
            x0=[1.4, 1, 6.3],
            init_args=[2],
            vary_par_idx=0,
            max_par=-1,
            num_steps=50,
            discretisation=numerical_shooting.shooting_problem,
            solver=sp.optimize.fsolve,
            phase_con=hopf_pc
        )
        self.assertEqual(np.shape(x_pred), (4, 50))
        self.assertAlmostEqual(round(x_pred[0][-1], 3), -0.570)
        self.assertAlmostEqual(round(x_pred[1][-1], 3), 0.003)
        self.assertAlmostEqual(round(x_pred[2][-1], 3), 6.252)
        self.assertAlmostEqual(round(x_pred[3][-1], 3), 0.320)


class test_methods_pde_solver(unittest.TestCase):

    def test_solve_diffusion(self):
        
        # DIRICHLET BOUNDARY TYPE
        x_pred = pde_solver.solve_diffusion(
            method='explicit_euler', 
            boundary_type='dirichlet', 
            l_bound_func=diff_l_bound, 
            r_bound_func=diff_r_bound, 
            r_bound_args=[0, 0],
            init_func=diff_init, 
            D=1, 
            x_min=0, 
            x_max=5, 
            nx=100, 
            t_min=0, 
            t_max=0.5, 
            nt=1000, 
            init_args=[0, 5])
        self.assertEqual(np.shape(x_pred), (2, 101))
        self.assertEqual(x_pred[0][0], 0)
        self.assertEqual(x_pred[0][-1], 0)
        self.assertAlmostEqual(round(x_pred[0][10], 3), 0.254)
        self.assertAlmostEqual(round(x_pred[0][-10], 3), 0.229)
        self.assertAlmostEqual(round(x_pred[1][10], 3), 0.500)
        self.assertAlmostEqual(round(x_pred[1][-10], 3), 4.550)

        x_pred = pde_solver.solve_diffusion(
            method='implicit_euler', 
            boundary_type='dirichlet', 
            l_bound_func=diff_l_bound, 
            r_bound_func=diff_r_bound, 
            r_bound_args=[0, 0],
            init_func=diff_init, 
            D=1, 
            x_min=0, 
            x_max=5, 
            nx=100, 
            t_min=0, 
            t_max=0.5, 
            nt=1000, 
            init_args=[0, 5])
        self.assertEqual(np.shape(x_pred), (2, 101))
        self.assertEqual(x_pred[0][0], 0)
        self.assertEqual(x_pred[0][-1], 0)
        self.assertAlmostEqual(round(x_pred[0][10], 3), 0.254)
        self.assertAlmostEqual(round(x_pred[0][-10], 3), 0.229)
        self.assertAlmostEqual(round(x_pred[1][10], 3), 0.500)
        self.assertAlmostEqual(round(x_pred[1][-10], 3), 4.550)

        x_pred = pde_solver.solve_diffusion(
            method='crank_nicolson', 
            boundary_type='dirichlet', 
            l_bound_func=diff_l_bound, 
            r_bound_func=diff_r_bound, 
            r_bound_args=[0, 0],
            init_func=diff_init, 
            D=1, 
            x_min=0, 
            x_max=5, 
            nx=100, 
            t_min=0, 
            t_max=0.5, 
            nt=1000, 
            init_args=[0, 5])
        self.assertEqual(np.shape(x_pred), (2, 101))
        self.assertEqual(x_pred[0][0], 0)
        self.assertEqual(x_pred[0][-1], 0)
        self.assertAlmostEqual(round(x_pred[0][10], 3), 0.254)
        self.assertAlmostEqual(round(x_pred[0][-10], 3), 0.229)
        self.assertAlmostEqual(round(x_pred[1][10], 3), 0.500)
        self.assertAlmostEqual(round(x_pred[1][-10], 3), 4.550)


        # NEUMANN BOUNDARY TYPE
        x_pred = pde_solver.solve_diffusion(
            method='explicit_euler', 
            boundary_type='neumann', 
            l_bound_func=diff_l_bound, 
            r_bound_func=diff_r_bound, 
            r_bound_args=[0, 0],
            init_func=diff_init, 
            D=1, 
            x_min=0, 
            x_max=5, 
            nx=100, 
            t_min=0, 
            t_max=0.5, 
            nt=1000, 
            init_args=[0, 5])
        self.assertEqual(np.shape(x_pred), (2, 101))
        self.assertEqual(x_pred[0][0], 0)
        self.assertAlmostEqual(round(x_pred[0][-1], 3), 0.440)
        self.assertAlmostEqual(round(x_pred[0][10], 3), 0.254)
        self.assertAlmostEqual(round(x_pred[0][-10], 3), 0.472)
        self.assertAlmostEqual(round(x_pred[1][10], 3), 0.500)
        self.assertAlmostEqual(round(x_pred[1][-10], 3), 4.550)

        x_pred = pde_solver.solve_diffusion(
            method='implicit_euler', 
            boundary_type='neumann', 
            l_bound_func=diff_l_bound, 
            r_bound_func=diff_r_bound, 
            r_bound_args=[0, 0],
            init_func=diff_init, 
            D=1, 
            x_min=0, 
            x_max=5, 
            nx=100, 
            t_min=0, 
            t_max=0.5, 
            nt=1000, 
            init_args=[0, 5])
        self.assertEqual(np.shape(x_pred), (2, 101))
        self.assertEqual(x_pred[0][0], 0)
        self.assertAlmostEqual(round(x_pred[0][-1], 3), 0.440)
        self.assertAlmostEqual(round(x_pred[0][10], 3), 0.254)
        self.assertAlmostEqual(round(x_pred[0][-10], 3), 0.472)
        self.assertAlmostEqual(round(x_pred[1][10], 3), 0.500)
        self.assertAlmostEqual(round(x_pred[1][-10], 3), 4.550)

        x_pred = pde_solver.solve_diffusion(
            method='crank_nicolson', 
            boundary_type='neumann', 
            l_bound_func=diff_l_bound, 
            r_bound_func=diff_r_bound, 
            r_bound_args=[0, 0],
            init_func=diff_init, 
            D=1, 
            x_min=0, 
            x_max=5, 
            nx=100, 
            t_min=0, 
            t_max=0.5, 
            nt=1000, 
            init_args=[0, 5])
        self.assertEqual(np.shape(x_pred), (2, 101))
        self.assertEqual(x_pred[0][0], 0)
        self.assertAlmostEqual(round(x_pred[0][-1], 3), 0.440)
        self.assertAlmostEqual(round(x_pred[0][10], 3), 0.254)
        self.assertAlmostEqual(round(x_pred[0][-10], 3), 0.472)
        self.assertAlmostEqual(round(x_pred[1][10], 3), 0.500)
        self.assertAlmostEqual(round(x_pred[1][-10], 3), 4.550)


        # ROBIN BOUNDARY TYPE
        x_pred = pde_solver.solve_diffusion(
            method='explicit_euler', 
            boundary_type='robin', 
            l_bound_func=diff_l_bound, 
            r_bound_func=diff_r_bound, 
            r_bound_args=[0, 2],
            init_func=diff_init, 
            D=1, 
            x_min=0, 
            x_max=5, 
            nx=100, 
            t_min=0, 
            t_max=0.5, 
            nt=1000, 
            init_args=[0, 5])
        self.assertEqual(np.shape(x_pred), (2, 101))
        self.assertEqual(x_pred[0][0], 0)
        self.assertAlmostEqual(round(x_pred[0][-1], 3), 0.178)
        self.assertAlmostEqual(round(x_pred[0][10], 3), 0.254)
        self.assertAlmostEqual(round(x_pred[0][-10], 3), 0.339)
        self.assertAlmostEqual(round(x_pred[1][10], 3), 0.500)
        self.assertAlmostEqual(round(x_pred[1][-10], 3), 4.550)

        x_pred = pde_solver.solve_diffusion(
            method='implicit_euler', 
            boundary_type='robin', 
            l_bound_func=diff_l_bound, 
            r_bound_func=diff_r_bound, 
            r_bound_args=[0, 2],
            init_func=diff_init, 
            D=1, 
            x_min=0, 
            x_max=5, 
            nx=100, 
            t_min=0, 
            t_max=0.5, 
            nt=1000, 
            init_args=[0, 5])
        self.assertEqual(np.shape(x_pred), (2, 101))
        self.assertEqual(x_pred[0][0], 0)
        self.assertAlmostEqual(round(x_pred[0][-1], 3), 0.178)
        self.assertAlmostEqual(round(x_pred[0][10], 3), 0.254)
        self.assertAlmostEqual(round(x_pred[0][-10], 3), 0.339)
        self.assertAlmostEqual(round(x_pred[1][10], 3), 0.500)
        self.assertAlmostEqual(round(x_pred[1][-10], 3), 4.550)

        x_pred = pde_solver.solve_diffusion(
            method='crank_nicolson', 
            boundary_type='robin', 
            l_bound_func=diff_l_bound, 
            r_bound_func=diff_r_bound, 
            r_bound_args=[0, 2],
            init_func=diff_init, 
            D=1, 
            x_min=0, 
            x_max=5, 
            nx=100, 
            t_min=0, 
            t_max=0.5, 
            nt=1000, 
            init_args=[0, 5])
        self.assertEqual(np.shape(x_pred), (2, 101))
        self.assertEqual(x_pred[0][0], 0)
        self.assertAlmostEqual(round(x_pred[0][-1], 3), 0.178)
        self.assertAlmostEqual(round(x_pred[0][10], 3), 0.254)
        self.assertAlmostEqual(round(x_pred[0][-10], 3), 0.339)
        self.assertAlmostEqual(round(x_pred[1][10], 3), 0.500)
        self.assertAlmostEqual(round(x_pred[1][-10], 3), 4.550)


# RUN TESTS
if __name__ == '__main__':
    unittest.main()

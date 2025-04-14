from scipy.integrate import solve_ivp
import numpy as np

def simulate(field, x0, T, dt):
    t_eval = np.arange(0, T, dt)
    sol = solve_ivp(field, [0, T], x0, t_eval=t_eval)
    return sol.y.T

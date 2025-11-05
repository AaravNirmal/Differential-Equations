import numpy as np

from integrator import rk_solver

class RungeKutta4Integrator:
    
    def __init__(self, dt):
        self.dt = dt
        
    def solve(self, field_object, x0, T):
        trajectory = rk_solver.solve(
            field_object, 
            x0, 
            T, 
            self.dt
        )
        return trajectory
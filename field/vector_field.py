import numpy as np
import sympy as sp

class VectorField:
    def __init__(self, expressed_string, params=None):
        self.n = len(expressed_string)
        self.state_vars = [sp.Symbol(f'x{i}') for i in range(self.n)] 
        all_symbols = self.state_vars.copy()
        parsed_expressions = [sp.sympify(ex) for ex in expressed_string]  
        self.derivative = sp.Matrix(parsed_expressions)

        self.compiled_function = sp.lambdify(all_symbols, self.derivative, modules='numpy', dummify=True)  

    def __call__(self, x, t):
        return np.squeeze(self.compiled_function(*x))
        

import numpy as np
import sympy as sp

def make_field_from_expr(exprs):
    n = len(exprs)
    vars = [sp.Symbol(f'x{i}') for i in range(n)]
    parsed_exprs = [sp.sympify(e) for e in exprs]
    lambdas = [sp.lambdify(vars, expr, modules='numpy') for expr in parsed_exprs]

    def field(t, x):
        return np.array([f(*x) for f in lambdas])
    return field

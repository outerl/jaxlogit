from jaxlogit._optimize import hessian
import numpy as np
import jax as jax


def test_hessian():
    # x is the betas
    def test_function(x, a, b, c, force_positive_chol_diag, num_panels):
        return a**2 + b**2 + a * b * c

    a = 5.0
    b = 2.0
    c = 3
    fpcd = True
    num_panels = 3
    args = (a, b, c, fpcd, num_panels)
    x = np.repeat(0.1, 3)
    hessian(test_function, x, False, False, *args)

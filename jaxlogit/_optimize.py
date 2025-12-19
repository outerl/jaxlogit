import logging

import jax
import jax.numpy as jnp


logger = logging.getLogger(__name__)

# static_argnames in loglikelihood function, TODO: maybe replace with partial and get rid of all additional args
STATIC_LOGLIKE_ARGNAMES = ["num_panels", "force_positive_chol_diag", "parameter_info"]


def _minimize(loglik_fn, x, args, method, tol, options, jit_loglik=True):
    logger.info(f"Running minimization with method {method}")
    if method in ["L-BFGS-B", "BFGS"]:
        from scipy.optimize import minimize

        if jit_loglik:
            neg_loglik_and_grad = jax.jit(
                jax.value_and_grad(loglik_fn, argnums=0), static_argnames=STATIC_LOGLIKE_ARGNAMES
            )
        else:
            # If we are batching, we provide both
            neg_loglik_and_grad = loglik_fn

        def neg_loglike_scipy(betas, *args):
            """Wrapper for neg_loglike to use with scipy."""
            x = jnp.array(betas)
            return neg_loglik_and_grad(x, *args)

        nit = 0  # global counter for display callback

        def display_callback(optim_res):
            nonlocal nit, neg_loglike_scipy, args
            nit += 1
            val, grad = neg_loglike_scipy(optim_res, *args)
            g_norm = jnp.linalg.norm(grad)
            logger.info(f"Iter {nit}, fun = {val:.3f}, |grad| = {g_norm:.3f}")  # , current sol = {optim_res}")

        if method == "L-BFGS-B":
            return minimize(
                neg_loglike_scipy,
                x,
                args=args,
                jac=True,
                method="L-BFGS-B",
                tol=tol,
                options=options,
            )
        elif method == "BFGS":
            return minimize(
                neg_loglike_scipy,
                x,
                args=args,
                jac=True,
                method="BFGS",
                options=options,
                callback=display_callback if options["disp"] else None,
            )
        else:
            logger.error(f"Unknown optimization method: {method} exiting gracefully")
            return None
    else:
        logger.error(f"Unknown optimization method: {method} exiting gracefully")
        return None


def gradient(funct, x, *args):
    """Finite difference gradient approximation."""

    # # memory intensive for large x and large sample sizes
    # grad = jax.jacobian(funct, argnums=0)(jnp.array(x), *args)
    # struggles with very small numbers

    # Finite differences, lowest memory usage but slowest
    eps = 1e-6
    n = x.size
    if n == 0:
        raise ValueError("x must have at least one dimension")
    grad_shape = funct(x, *args).size
    grad = jnp.zeros((grad_shape, n))
    for i in range(n):
        x_plus = x.at[i].set(x[i] + eps)
        x_minus = x.at[i].set(x[i] - eps)
        grad = grad.at[:, i].set((funct(x_plus, *args) - funct(x_minus, *args)) / (2 * eps))
    return grad


def hessian(funct, x, hessian_by_row, finite_diff, *args, static_argnames=STATIC_LOGLIKE_ARGNAMES):
    """Compute the Hessian of funct for variables x."""

    # # this is memory intensive for large x.
    # hess_fn = jax.jacfwd(jax.grad(funct))  # jax.hessian(neg_loglike)
    # H = hess_fn(jnp.array(x), *args)

    grad_funct = jax.jit(jax.grad(funct, argnums=0), static_argnames=static_argnames)

    # This is a compromise between memory and speed - we know jax gradient calculations are
    # within memory limits because we use it during minimization, to stay within the same
    # memory limits we use finite differences on the jitted grad.
    if finite_diff:
        eps = 1e-6
        H = jnp.empty((len(x), len(x)))
        for i in range(len(x)):
            x_plus = x.at[i].set(x[i] + eps)
            x_minus = x.at[i].set(x[i] - eps)
            grad_plus = grad_funct(x_plus, *args)  # gradient(funct, x_plus, *args) for full FD
            grad_minus = grad_funct(x_minus, *args)
            hess_row = (grad_plus - grad_minus) / (2 * eps)
            H = H.at[i, :].set(hess_row)
    else:

        def row(i):
            return jax.grad(lambda x_: grad_funct(x_, *args)[i])(x)

        if not hessian_by_row:
            H = jax.vmap(row)(jnp.arange(x.size))
        # slower but less memory intensive - hessian_by_row in for loop
        else:
            H = jnp.empty((len(x), len(x)))
            for i in range(len(x)):
                hess_row = row(i)
                H = H.at[i, :].set(hess_row)

    return H

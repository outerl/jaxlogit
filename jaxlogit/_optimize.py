import logging

import jax
import jax.numpy as jnp
import jaxopt
import optax
import optax.tree

from scipy.optimize import minimize, OptimizeResult
from jax.scipy.optimize import minimize as jminimize
from jax.scipy.optimize import OptimizeResults as OR

from functools import partial


logger = logging.getLogger(__name__)

# static_argnames in loglikelihood function, TODO: maybe replace with partial and get rid of all additional args
STATIC_LOGLIKE_ARGNAMES = ["num_panels", "force_positive_chol_diag", "parameter_info"]


def jit_optax_func(loglik_fn, args):
    (Xdf, Xdr, panels, weights, avail, num_panels, force_positive_chol_diag, draws, parameter_info, batch_size) = args

    @partial(jax.jit, static_argnames=STATIC_LOGLIKE_ARGNAMES)
    def func(
        x, *, Xdf, Xdr, panels, weights, avail, draws, num_panels, force_positive_chol_diag, parameter_info, batch_size
    ):
        return loglik_fn(
            x, Xdf, Xdr, panels, weights, avail, num_panels, force_positive_chol_diag, draws, parameter_info, batch_size
        )

    def wrapper(x):
        return func(
            x,
            Xdf=Xdf,
            Xdr=Xdr,
            panels=panels,
            weights=weights,
            avail=avail,
            draws=draws,
            num_panels=num_panels,
            force_positive_chol_diag=force_positive_chol_diag,
            parameter_info=parameter_info,
            batch_size=batch_size,
        )

    return wrapper


def optax_minimize(loglik_fn, x, args, options, tol):
    func = jit_optax_func(loglik_fn, args)

    solver = optax.lbfgs()
    opt_state = solver.init(x)
    value_and_grad = optax.value_and_grad_from_state(func)
    success = False
    last_x = x

    for _ in range(options["maxiter"]):
        value, grad = value_and_grad(x, state=opt_state)
        updates, opt_state = solver.update(grad, opt_state, x, value=value, grad=grad, value_fn=func)
        x = optax.apply_updates(x, updates)
        iter_num = optax.tree.get(opt_state, "count")
        grad = optax.tree.get(opt_state, "grad")
        err = optax.tree.norm(grad)
        if err < options["gtol"]:
            success = True
            break
        if abs(x - last_x).any() < tol:
            success = True
            break
        last_x = x
    return OR(x, success, 0, value, None, None, None, None, iter_num)


def scipy_result_to_jax(result: OptimizeResult):
    return OR(
        result["x"],
        result["success"],
        result["status"],
        result["fun"],
        result["jac"],
        result["hess_inv"],
        result["nfev"],
        result["njev"],
        result["nit"],
    )


def _minimize(loglik_fn, x, args, method, tol, options, jit_loglik=True):
    logger.info(f"Running minimization with method {method}")
    neg_loglik_and_grad = loglik_fn
    x = jnp.array(x)

    def neg_loglike_scipy(betas, *args):
        """Wrapper for neg_loglike to use with scipy."""
        # x = jnp.array(betas)
        return neg_loglik_and_grad(x, *args)

    if method == "L-BFGS-jax":
        return jminimize(
            loglik_fn,
            jnp.array(x),
            args=args,
            method="l-bfgs-experimental-do-not-rely-on-this",
            tol=tol,
            options=options,
        )
    elif method == "BFGS-jax":
        return jminimize(
            neg_loglike_scipy,
            jnp.array(x),
            args=args,
            method="BFGS",
            tol=tol,
            options=options,
        )
    elif method == "L-BFGS-B-jaxopt":  # Not fully working yet
        if jit_loglik:
            neg_loglik_and_grad = jax.jit(
                jax.value_and_grad(loglik_fn, argnums=0), static_argnames=STATIC_LOGLIKE_ARGNAMES
            )
        else:
            # If we are batching, we provide both
            neg_loglik_and_grad = loglik_fn

        bounds = (
            jnp.full_like(x, -jnp.inf),
            jnp.full_like(x, jnp.inf),
        )

        obj = jaxopt.LBFGSB(neg_loglik_and_grad, value_and_grad=True, maxiter=options["maxiter"])
        result = obj.run(x, bounds, *args)
        return result
    elif method == "L-BFGS-optax":
        return optax_minimize(loglik_fn, x, args, options, tol)
    elif method == "L-BFGS-B-scipy":
        if jit_loglik:
            neg_loglik_and_grad = jax.jit(
                jax.value_and_grad(loglik_fn, argnums=0), static_argnames=STATIC_LOGLIKE_ARGNAMES
            )
        else:
            # If we are batching, we provide both
            neg_loglik_and_grad = loglik_fn
        return scipy_result_to_jax(
            minimize(
                neg_loglike_scipy,
                x,
                args=args,
                jac=True,
                method="L-BFGS-B",
                tol=tol,
                options=options,
            )
        )
    elif method == "BFGS-scipy":
        if jit_loglik:
            neg_loglik_and_grad = jax.jit(
                jax.value_and_grad(loglik_fn, argnums=0), static_argnames=STATIC_LOGLIKE_ARGNAMES
            )
        else:
            # If we are batching, we provide both
            neg_loglik_and_grad = loglik_fn
        return scipy_result_to_jax(
            minimize(
                neg_loglike_scipy,
                x,
                args=args,
                jac=True,
                method="BFGS",
                tol=tol,
                options=options,
            )
        )

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

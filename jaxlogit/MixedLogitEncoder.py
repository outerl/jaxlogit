from json import JSONEncoder
import numpy as np
import jax.numpy as jnp
from jaxlogit.mixed_logit import MixedLogit


class MixedLogitEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) or isinstance(obj, jnp.ndarray):
            return obj.tolist()
        return obj.__dict__


def mixed_logit_decoder(obj):
    try:
        model = MixedLogit()
        model.coeff_names = np.array(obj["coeff_names"])
        model.coeff_ = np.array(obj["coeff_"])
        model.stderr = np.array(obj["stderr"])
        model.zvalues = np.array(obj["zvalues"])
        model.loglikelihood = obj["loglikelihood"]
        model.aic = obj["aic"]
        model.bic = obj["bic"]
        return model
    except KeyError:
        return obj

def optim_res_decoder(obj):
    try:
        model = MixedLogit()
        model.estimation_message = np.array(obj["estimation_message"])
        model.success = np.array(obj["success"])
        model.status = np.array(obj["status"])
        model.fun = np.array(obj["fun"])
        model.x = obj["x"]
        model.nit = obj["nit"]
        model.jac = obj["jac"]
        model.nfev = obj["nfev"]
        model.njev = obj["njev"]
        return model
    except KeyError:
        return obj

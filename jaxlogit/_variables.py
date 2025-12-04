class Variables:
    """
    A grouping of commonly used variables

    #TODO: see whether any functions can be abstracted here
    #TODO: docstring description of each
    #TODO: add class logic
    """

    def __init__(
        self,
        Xdf,
        Xdr,
        draws,
        mask,
        values_for_mask,
        mask_chol,
        values_for_chol_mask,
        rand_idx_norm,
        rand_idx_truncnorm,
        draws_idx_norm,
        draws_idx_truncnorm,
        fixed_idx,
        idx_ln_dist,
        rand_idx_stddev,
        rand_idx_chol,
    ):
        self.Xdf = Xdf
        self.Xdr = Xdr
        self.draws = draws
        self.mask = mask
        self.values_for_mask = values_for_mask
        self.mask_chol = mask_chol
        self.values_for_chol_mask = values_for_chol_mask
        self.rand_idx_norm = rand_idx_norm
        self.rand_idx_truncnorm = rand_idx_truncnorm
        self.draws_idx_norm = draws_idx_norm
        self.draws_idx_truncnorm = draws_idx_truncnorm
        self.fixed_idx = fixed_idx
        self.idx_ln_dist = idx_ln_dist
        self.rand_idx_stddev = rand_idx_stddev
        self.rand_idx_chol = rand_idx_chol

import numpy as np


def sample_parameters(var_1_ind, var_2_ind, means, sds, cor, N):
    """
    Draw N bettors' attributes (budget, risk aversion, stubbornness, expertise) from a
    multivariate normal with the given means and standard deviations, where the two chosen
    attributes have correlation `cor` and all other pairs are independent.
    """
    cov_matrix = np.diag(sds) ** 2
    cov_ = cor * sds[var_1_ind] * sds[var_2_ind]
    cov_matrix[var_1_ind, var_2_ind] = cov_
    cov_matrix[var_2_ind, var_1_ind] = cov_
    return np.random.multivariate_normal(means, cov_matrix, N)

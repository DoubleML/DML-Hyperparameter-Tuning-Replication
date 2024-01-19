import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.linalg import toeplitz

def DGP_BCH2014(theta = 0.5, n_obs = 100, dim_x = 200, rho = 0.5,
                R2_d = 0.5, R2_y = 0.5, design = '1a'):
    """
    Generated data from a partially linear regression model as of Belloni et al. (2014).    
    Implements designs 1(a) and 2(a)
        
    # TODO: Add documentation
    # TODO: Asserts for inputs
    """

    v = np.random.standard_normal(size=[n_obs, ])
    zeta = np.random.standard_normal(size=[n_obs, ])
    cov_mat = toeplitz([np.power(rho, k) for k in range(dim_x)])
    x = np.random.multivariate_normal(np.zeros(dim_x), cov_mat, size=[n_obs, ])

    if design == '1a':
        beta_y = np.concatenate((1/np.arange(1,6), np.zeros(5),
                                1/np.arange(1,6), np.zeros(dim_x - 15)))
        beta_d = np.concatenate((1/np.arange(1,11), np.zeros(dim_x - 10)))
        
    if design == '2a':
        beta_y= np.concatenate((1/np.power(np.arange(1,6),2), np.zeros(5),
                                1/np.power(np.arange(1,6),2), np.zeros(dim_x - 15)))
        beta_d = np.concatenate((1/np.power(np.arange(1,11),2), np.zeros(dim_x - 10)))

    b_y_sigma_b_y = np.dot(np.dot(cov_mat, beta_y), beta_y)
    b_d_sigma_b_d = np.dot(np.dot(cov_mat, beta_d), beta_d)

    c_y = np.sqrt(R2_y/((1-R2_y) * b_y_sigma_b_y))
    c_d = np.sqrt(R2_d/((1-R2_d) * b_d_sigma_b_d))

    d = np.dot(x, np.multiply(beta_d, c_d)) + v
    y = d * theta + np.dot(x, np.multiply(beta_y, c_y)) + zeta
    
    true_betas = {'beta_y': np.multiply(beta_y, c_y), 'beta_d': np.multiply(beta_d, c_d)}
    
    # for ml_l: use partialled out y for evaluation
    # TODO: Fix orcl RMSE for ml_l
    y_pred_orcl = y - zeta
    d_pred_orcl = d - v
    orcl_rmse_y = mean_squared_error(y, y_pred_orcl, squared=False)
    orcl_rmse_d = mean_squared_error(d, d_pred_orcl, squared=False)
    
    orcl_rmse = {'rmse_y': orcl_rmse_y,
                 'rmse_d': orcl_rmse_d}
    
    dgp_info = {'R2_y': R2_y,
                'R2_d': R2_d,
                'rho': rho,
                'design': design}
    
    return x, y, d #, true_betas, orcl_rmse, dgp_info
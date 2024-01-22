import pandas as pd
import numpy as np
import doubleml as dml
from doubleml import DoubleMLData
from doubleml._utils import _rmse

from sklearn.linear_model import Lasso
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error

from datetime import date

from itertools import product

from scipy.linalg import toeplitz

from doubleml import DoubleMLData

import warnings
warnings.filterwarnings('ignore', category = ConvergenceWarning)


def DGP_BCH2014(theta = 0.5, n_obs = 100, dim_x = 200, rho = 0.5,
                R2_d = 0.5, R2_y = 0.5, design = '1a'):
    """
    Generated data from a partially linear regression model as of Belloni et al. (2014).    
    Implements designs 1(a) and 2(a)
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
    y_pred_orcl = y - zeta
    d_pred_orcl = d - v
    orcl_rmse_y = _rmse(y, y_pred_orcl)
    orcl_rmse_d = _rmse(d, d_pred_orcl)
    
    orcl_rmse = {'rmse_y': orcl_rmse_y,
                 'rmse_d': orcl_rmse_d}
    
    dgp_info = {'R2_y': R2_y,
                'R2_d': R2_d,
                'rho': rho,
                'design': design}
    
    return x, y, d, true_betas, orcl_rmse, dgp_info

def cover_true(theta, confint):
    """

    Function to check whether theta is contained in confindence interval.
    Returns 1 if true and 0 otherwise.
    
    """
    covers_theta = (confint[0] < theta and theta < confint[1])
    
    if covers_theta:
        return 1
    else:
        return 0


class ResultsOneRep:
    def __init__(self, dml_obj = None, theta = None, level = 0.95, alpha_l = -1,
                 alpha_m = -1, i_rep = -1, orcl_rmse = None, dgp_info = None):
        self.theta = theta
        self.rho = dgp_info['rho']
        self.R2_y = dgp_info['R2_y']
        self.R2_d = dgp_info['R2_d']
        self.design = dgp_info['design']
        self.dgp_info = dgp_info

        self.coef = dml_obj.coef[0]
        self.se = dml_obj.se[0]
        self.level = level
        self.confint =  dml_obj.confint(level = level).iloc[0]
        self.abs_bias = np.abs(self.coef - theta)
        self.sq_error = np.power(self.coef - theta, 2)
        self.cover = cover_true(theta, self.confint)
        self.ci_width = self.confint[1] - self.confint[0]
        
        self.nuis_rmse = dml_obj.evaluate_learners()
        self.orcl_rmse = orcl_rmse
        self.alpha_ml_l = alpha_l
        self.alpha_ml_m = alpha_m
        self.i_rep = i_rep
        
        self.date = str(date.today())
        
        df_results = pd.DataFrame.from_dict({'theta': [self.theta],
                                          'coef': [self.coef],
                                          'se': [self.se],
                                          'confint_l': [self.confint[0]],
                                          'confint_u': [self.confint[1]],
                                          'abs_bias': [self.abs_bias],
                                          'sq_error': [self.sq_error],
                                          'cover': [self.cover],
                                          'level': [self.level],
                                          'ci_width': [self.ci_width],
                                          'nuis_rmse_ml_l': [self.nuis_rmse['ml_l'][0][0]],
                                          'nuis_rmse_ml_m': [self.nuis_rmse['ml_m'][0][0]],
                                          'orcl_rmse_ml_l': [self.orcl_rmse['rmse_y']],
                                          'orcl_rmse_ml_m': [self.orcl_rmse['rmse_d']],
                                          'alpha_ml_l' : [self.alpha_ml_l],
                                          'alpha_ml_m' : [self.alpha_ml_m],
                                          'i_rep': [self.i_rep],
                                          'date': [self.date],
                                          'rho': [self.rho],
                                          'R2_y': [self.R2_y],
                                          'R2_d': [self.R2_d],
                                          'design': [self.design]})
        
        self.df_results = df_results
        
    def __str__(self):
        str_summary = f"Coef: {self.coef}\n" \
                      f"Abs. bias: {self.abs_bias}\n" \
                      f"Covers true theta: {self.cover}\n" \
                      f"Conf. interval width: {self.ci_width}\n" \
                      f"rmse nuisance: {self.nuis_rmse}\n" \
                      f"orcle rmse nuisance: {self.orcl_rmse}\n"\
                      f"alpha ml_l: {self.alpha_ml_l}\n" \
                      f"alpha ml_m: {self.alpha_ml_m}\n" \
                      f"dgp info: {self.dgp_info}"
        
        return str(str_summary)

    def store_results(self, path):
        """
        Store results from single repetition to .csv file
        """
        self.df_results.to_csv(path)

def run_one_rep(i_rep = -1, alphas_ml_l = [-1], alphas_ml_m = [-1], n_vars = 200,
                n_obs = 100, theta = 0.5, data = None, level = 0.95):
    """
    Function executing one repetition of the simulation
    
    Args:
        - i_rep
        - lambda_l
        - lambda_m
        
    """
    
    combi_list = product(alphas_ml_l, alphas_ml_m)
    
    (x, y, d, true_betas, orcl_rmse, dgp_info) = data[0]
    alpha_results = []
    df_results = []
    
    for this_alpha in combi_list:
        alpha_l = this_alpha[0]
        alpha_m = this_alpha[1]
        ml_l = Lasso(alpha = alpha_l, random_state = 1)
        ml_m = Lasso(alpha = alpha_m, random_state = 1)

        # Fix seed to abstract from randomness in sample split
        np.random.seed(42 + i_rep)
        dml_data = DoubleMLData.from_arrays(x, y, d)
        dml_plr_obj = dml.DoubleMLPLR(dml_data, ml_l, ml_m, n_folds = 10)
        dml_plr_obj.fit(store_models = True, store_predictions = True)
        this_alpha_result = ResultsOneRep(dml_plr_obj, theta, level, alpha_l, alpha_m, i_rep,
                                          orcl_rmse, dgp_info)
        alpha_results.append(this_alpha_result)
        df_results.append(this_alpha_result.df_results)
    
    df_results_out = pd.concat(df_results, ignore_index = True)
        
    return df_results_out


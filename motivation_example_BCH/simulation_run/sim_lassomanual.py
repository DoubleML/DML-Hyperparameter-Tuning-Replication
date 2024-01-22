# Imports
import doubleml as dml
from doubleml import DoubleMLData

import numpy as np
import pandas as pd

from itertools import product

# Import DGP from local module
from importlib.machinery import SourceFileLoader

# imports the module from the given path
# import simulationlasso as sml
sml = SourceFileLoader("simulationlasso","../code/simulationlasso.py").load_module()

from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV

import matplotlib.pyplot as plt

# Parameters
import time

# Parametrize Simulation
n_vars = 200
n_obs = 100
n_rep = 100
theta = 0.5
level = 0.95

# Set up different settings
rho = [0.2, 0.6, 0.8]
R2_y = [0.2, 0.6, 0.8]
R2_d = [0.2, 0.6, 0.8]

design = ["1a", "2a"]

np.random.seed(1234)
# Set up settings

# First: set same value for R2_d and R2_y
settings = product(rho, R2_y, design)
settings_list = list(product(rho, R2_y, design))
print(settings_list)
n_sett = len(rho) * len(R2_y) * len(design)

setting_counter = 1
for i_set in settings:
    
    this_rho = i_set[0]
    this_R2 = i_set[1]
    this_design = i_set[2]
    
    # Simulate data

    # Prepare objects
    data_list = [(-1, -1, -1, -1, -1, -1)]*n_rep
    df_results = [None] * n_rep

    for i_rep in range(n_rep):
        (x, y, d, true_betas, orcl_rmse, dgp_info) = sml.DGP_BCH2014(dim_x = n_vars, n_obs = n_obs, theta = theta,
                                                            rho = this_rho, R2_d = this_R2, R2_y = this_R2, design = this_design)
        this_data = [(x, y, d, true_betas, orcl_rmse, dgp_info)]
        data_list[i_rep] = this_data

    # Execution time
    start = time.time()

    for i_rep in range(n_rep):
                
        # Use LassoCV for first repetition to get initial list of alphas for each setting
        if i_rep == 0:
            print(f"Simulation setting with rho = {this_rho} and R2 = {this_R2} , design {this_design},\n"
                  f"setting {setting_counter}/{n_sett}")
            # ml_l
            (x, y, d, true_betas, orcl_rmse, dgp_info) = data_list[i_rep][0]
            fit_l = LassoCV(cv = 10, random_state = 42)
            fit_l.fit(x,y)
            alphas_ml_l = fit_l.alphas_
            
            # ml_m
            fit_m = LassoCV(cv = 10, random_state = 42)
            fit_m.fit(x,d)
            alphas_ml_m = fit_m.alphas_
            
            # alpha grid (every 10th value)
            alphas_ml_l_in = alphas_ml_l[::10]
            alphas_ml_m_in = alphas_ml_m[::10]
            
        
        one_rep = sml.run_one_rep(i_rep = i_rep, alphas_ml_l = alphas_ml_l_in, alphas_ml_m = alphas_ml_m_in,
                                  n_vars = n_vars, n_obs = n_obs, theta = theta, data = data_list[i_rep])
        df_results[i_rep] = one_rep

    setting_counter = setting_counter + 1
    
    end = time.time()

    duration = end - start
    print(f"Duration for this setting: {duration/60} minutes")

    df_results = pd.concat(df_results, ignore_index = True)
    # save results to csv
    df_results.to_csv(f"results/raw_res_manual_lasso_R_{n_rep}_n_{n_obs}_p_{n_vars}_rho_{this_rho}_R2d_{this_R2}_R2y_{this_R2}_design_{this_design}.csv")

print("Done!")

# Time: 10 x 10 grid with 100 repetitions - ca. 45 minutes
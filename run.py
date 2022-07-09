# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 17:30:02 2022

@author: skia_
"""
from funcs import *

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

from pathlib import Path

import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(levelname)s %(asctime)s - %(message)s")

from funcs import *
# %% Question 1
# N = 10000  # number of samples, must be an int
# rng = np.random.default_rng(seed=12345)
# mu, sigma = 1.06, np.sqrt(0.025656229)
# s = rng.lognormal(mu, sigma, N)
# count, bins, _ = plt.hist(s, 100, density=True)
# x = np.linspace(min(bins), max(bins), 10000)
# pdf = lognorm(s=sigma, scale=np.exp(mu)).pdf(x)
# plt.plot(x, pdf, linewidth=2, color='r')
# plt.axis('tight')
# plt.show()

# %% Question 2
fld_data = Path("./data")
file_name_list = ['covarianceMAtrix.csv',
                  'factorLoadings.csv',
                  'factors.csv',
                  'PortfolioHoldings.csv']
cov_df, factor_loadings_df, factors_df, ptf_holdings_df = (read_data(fld_data, x) for x in file_name_list)
sigma_f, omega, beta, w = parse_data(cov_df, factor_loadings_df, factors_df, ptf_holdings_df)
sigma_n = compute_sigma_n(sigma_f, beta, omega)
ptf_vola = compute_ptf_vola(sigma_n, beta, w)

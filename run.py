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
fld_data = Path("./data")
## Question 1.1 ---------------------------------------------------------------
m = 1.06
v = 0.025656229
mu, sigma_2 = compute_normal_params(m, v)
sigma = np.sqrt(sigma_2)
m_star = np.exp(mu)
m_hat, v_hat = check_logn_params(mu = mu, sigma = sigma)

## Question 1.2 ---------------------------------------------------------------
N = 10000  # number of samples, must be an int
rng = np.random.default_rng(seed=12345)
s = rng.lognormal(mu, sigma, N)
count, bins, _ = plt.hist(s, 100, density=True)
x = np.linspace(min(bins), max(bins), 10000)
pdf = lognorm(s=sigma, scale=np.exp(mu)).pdf(x)
plt.plot(x, pdf, linewidth=2, color='r')
plt.axis('tight')
plt.show()
plt.savefig(fld_data / 'lognorm_hist.png')
# %% Question 2
## Get and parse data ---------------------------------------------------------
file_name_list = ['covarianceMAtrix.csv',
                  'factorLoadings.csv',
                  'factors.csv',
                  'PortfolioHoldings.csv']
cov_df, factor_loadings_df, factors_df, ptf_holdings_df = (read_data(fld_data, x) for x in file_name_list)
sigma_f, omega, beta, w = parse_data(cov_df, factor_loadings_df, factors_df, ptf_holdings_df)

## Question 2.1 --------------------------------------------------------------
sigma_n = compute_sigma_n(sigma_f, beta, omega)
ptf_vola = compute_ptf_vola(sigma_n, w)

## Question 2.2 ---------------------------------------------------------------
MRC = compute_MRC(sigma_n, w, ptf_vola)
RC = compute_RC(MRC, w)
PRC = compute_PRC(RC, ptf_vola)

ptf_holdings_df['PRC'] = PRC
ptf_holdings_df['RC'] = RC
ptf_holdings_df['PRC'] = PRC

plot_bar_chart(ptf_holdings_df, 
               'Weight',
               'PRC',
               fld_data,
               by_sector = False)
ptf_holdings_df_grouped = ptf_holdings_df[['Sector','Weight', 'PRC']].groupby('Sector').sum()
plot_bar_chart(ptf_holdings_df_grouped, 
               'Weight',
               'PRC',
               fld_data,
               by_sector = True)

## Question 2.3 ---------------------------------------------------------------
res = compute_ERC_weight(sigma_n, w)
w_ERC = res.x

ptf_vola_ERC = compute_ptf_vola(sigma_n, w_ERC)
MRC_ERC = compute_MRC(sigma_n, w_ERC, ptf_vola_ERC)
RC_ERC = compute_RC(MRC_ERC, w_ERC)
PRC_ERC = compute_PRC(RC_ERC, ptf_vola_ERC)

ptf_holdings_df['w_ERC'] = w_ERC
ptf_holdings_df['PRC_ERC'] = PRC_ERC
ptf_holdings_df['RC_ERC'] = RC_ERC
ptf_holdings_df['PRC_ERC'] = PRC_ERC

plot_bar_chart(ptf_holdings_df, 
               'w_ERC',
               'PRC_ERC',
               fld_data,
               by_sector = False)
#%% Latex table
to_latex_table(ptf_holdings_df, 
               ['Name', 'Weight', 'PRC'],
               fld_data,
               'table_by_asset.tex',
               'PRC',
               caption='Risk decomposition by asset')
to_latex_table(ptf_holdings_df_grouped, 
               ['Weight', 'PRC'],
               fld_data,
               'table_by_sector.tex',
               'PRC',
               caption='Risk decomposition by asset')
to_latex_table(ptf_holdings_df, 
               ['Name','w_ERC', 'PRC_ERC'],
               fld_data,
               'table_by_asset_ERC.tex',
               'PRC_ERC',
               caption='Risk decomposition by asset for the ERC portfolio')


# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 18:35:20 2022

@author: skia_
"""
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.optimize import minimize
from scipy.stats import lognorm

def compute_normal_params(m, v):
    sigma_2 = np.log(v / m**2 + 1)
    mu = np.log(m) - sigma_2 / 2
    return mu, sigma_2

def check_logn_params(mu, sigma):
    logn_dist = lognorm(s=sigma, scale=np.exp(mu))
    return logn_dist.mean(), logn_dist.var()
    

def read_data(fld_data, file_name):
    full_file_name = fld_data / file_name
    logging.info('Reading file = {}'.format(full_file_name))
    return pd.read_csv(full_file_name)

# convert the data from dataframe (useful for taking a look at it) to numpy
# arrays for computations
def parse_data(cov_df, factor_loadings_df, factors_df, ptf_holdings_df):
    logging.info('Parsing data')
    # we set the index in order to make sure data in consistent when converting
    # to numpy arrays
    factor_loadings_df.set_index('ISIN', inplace = True) 
    ptf_holdings_df.set_index('ISIN', inplace = True)
    assert(factor_loadings_df.index.equals(ptf_holdings_df.index))
    sigma_f = cov_df.drop('factors', axis = 1).to_numpy()
    # in the doc it's said specific variance (even though column is called volatility?)
    omega = (np.diag(factor_loadings_df.specificVolatility))
    beta = factor_loadings_df.drop(['specificVolatility'], axis = 1).to_numpy()
    w = ptf_holdings_df.Weight.to_numpy()
    return sigma_f, omega, beta, w

def compute_sigma_n(sigma_f, beta, omega):
    logging.info('Computing asset covariance matrix sigma_n')
    # according to the doc this is the formua to recover asset covariance matrix
    return np.sqrt(beta @ sigma_f @ beta.transpose() + omega) / 100

def compute_ptf_vola(sigma_n, w, flag_log = True):
    if flag_log: logging.info('Computing portfolio volatility')
    return np.sqrt(w.transpose() @ sigma_n @ w)
    
def compute_MRC(sigma_n, w, ptf_vola, flag_log = True):
    if flag_log: logging.info('Computing Marginal Risk Contribution (MRC)')
    return sigma_n @ w / ptf_vola

def compute_RC(MRC, w, flag_log = True):
    if flag_log: logging.info('Computing Risk Contribution (MRC)')
    return np.multiply(MRC, w)

def compute_PRC(RC, ptf_vola, flag_log = True):
    if flag_log: logging.info('Computing Percentage Risk Contribution (PRC)')
    return RC / ptf_vola


def plot_bar_chart(ptf_holdings_df, w_col, PRC_col, fld_data, by_sector = False):
    logging.info('Plotting pie char, by_sector = {}, w_col = {}, PRC_col = {}'.format(by_sector, w_col, PRC_col))
    if by_sector:
        labels = ptf_holdings_df.index.to_numpy()
        file_name = fld_data / 'bar_plot_by_sector_{}_{}.png'.format(w_col, PRC_col)
    else:
        labels = ptf_holdings_df['Name']
        file_name = fld_data / 'bar_plot_by_asset_{}_{}.png'.format(w_col, PRC_col)
    weight = ptf_holdings_df[w_col]
    PRC = ptf_holdings_df[PRC_col]
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.barh(x - width/2, weight * 100, width, label='Weights')
    rects2 = ax.barh(x + width/2, PRC * 100, width, label='PRC')
    ax.set_yticks(x)
    ax.set_yticklabels(labels)
    # ax.invert_yaxis()  
    ax.legend()
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)
    fig.tight_layout()
    plt.show()
    plt.savefig(file_name)
    
def obj_fun(w, sigma_n):
    risk_budget = 1 / sigma_n.shape[0]
    ptf_vola = compute_ptf_vola(sigma_n, w, flag_log = False)
    MRC = compute_MRC(sigma_n, w, ptf_vola, flag_log = False)
    RC = compute_RC(MRC, w, flag_log = False)
    PRC = compute_PRC(RC, ptf_vola, flag_log = False)
    return np.sum((PRC - risk_budget)**2) 

def compute_ERC_weight(sigma_n, w):
    logging.info('Computing ERC weights')
    n = sigma_n.shape[0]
    cons = {'type':'eq', 'fun': lambda x: np.sum(x) - 1}
    res = minimize(fun = obj_fun,
                   x0 = 1 / n,
                   args=(sigma_n),
                   method='SLSQP',
                   bounds=[(0, 1)] * n,
                   constraints=cons,
                   options={'disp': True})
    return res
    

def to_latex_table(ptf_holdings_df, cols_list, fld_data, file_name, sort_by, caption):
    ptf_holdings_df.sort_values(by=sort_by,
                                ascending=False).to_latex(buf=fld_data / file_name,
                                                          columns=cols_list,
                                                          float_format=lambda x: str(round(x * 100, 2)) + '%',
                                                          caption=caption,
                                                          position='H')


    
    
    
    
    


if __name__ == '__main__':
    pass

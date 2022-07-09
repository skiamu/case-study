# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 18:35:20 2022

@author: skia_
"""
import logging
import pandas as pd
import numpy as np


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
    w = ptf_holdings_df.Weight
    return sigma_f, omega, beta, w

def compute_sigma_n(sigma_f, beta, omega):
    logging.info('Computing asset covariance matrix sigma_n')
    # according to the doc this is the formua to recover asset covariance matrix
    return np.sqrt(beta @ sigma_f @ beta.transpose() + omega) / 100

def compute_ptf_vola(sigma_n, beta, w):
    logging.info('Computing portfolio volatility')
    return np.sqrt(w.transpose() @ sigma_n @ w)
    
    
    
    
    
    
    


if __name__ == '__main__':
    fld_data = Path("./data")

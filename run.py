# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 17:30:02 2022

@author: skia_
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
# from numpy.random import default_rng

#%% Question 1
N = 10000 # number of samples, must be an int
rng = np.random.default_rng(seed = 12345)
mu, sigma = 1.06, np.sqrt(0.025656229)
s = rng.lognormal(mu, sigma, N)
count, bins, _ = plt.hist(s, 100, density=True)
x = np.linspace(min(bins), max(bins), 10000)
pdf = lognorm(s = sigma, scale = np.exp(mu)).pdf(x)
plt.plot(x, pdf, linewidth=2, color='r')
plt.axis('tight')
plt.show()
#%% Question 2


if __name__ == '__main__':
    pass

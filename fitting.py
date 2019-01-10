""" This script provides methods to fit original data to several known 
distributions including Weibull, Rayleigh and lognormal.

There are two ways to find distribution parameters.
1. If probability data is available, the parameters can be found using
polynomial fitting. This is a nice alternative to scipy.stats methods.
2. If only original data is available or method 1 is not mathematically
applicable, use fitting methods provided by scipy.stats.

For Weibull fitting, two methods are provided, one can compare which fits
better by choosing 'param' and 'pdf' in line 49 and 50.
For Rayleigh fitting, method 1 is provided.
For lognormal fitting, method 2 is provided.
"""


import math
import numpy as np
from scipy import stats


def fit_distribution(prob_all, data_original, distribution):
    x_point = prob_all[0]
    prob_pdf = prob_all[1]
    prob_cdf = prob_all[2]
    num_duplicate = prob_all[3]
    # remove points with probability of 0 and 1 in case of using log
    x_point = x_point[:-1]
    prob_pdf = prob_pdf[:-1]
    prob_cdf = prob_cdf[:-1]
    num_duplicate = num_duplicate[:-1]

    num = len(x_point) + 1
    x_point_fit = np.linspace(0, max(x_point),num)
    if distribution == 'Weibull':
        # method 1: self-defined
        x = np.log(x_point)
        y = np.log(-np.log(1 - prob_cdf))
        p = np.polyfit(x, y, 1)
        k = p[0]
        lbd = np.exp(-p[1] / p[0])
        param_sd = [k, lbd]
        pdf_sd = ((k / lbd) * (x_point_fit / lbd) ** (k - 1)
                  * np.exp(-(x_point_fit / lbd) ** k))
        # method 2: scipy.stats method
        param_sp = stats.exponweib.fit(data_original)
        pdf_sp = stats.exponweib.pdf(x_point_fit, *param_sp)
        # choose return from the two methods and see which fits better
        param = param_sp  # param_sd or param_sp
        pdf = pdf_sp  # pdf_sd or pdf_sp
        label = r'{name} ($k$={k}, $\lambda$={lba})'.format(name=distribution, 
                  k=round(param[0], 2), lba=round(param[1], 2))
        return x_point_fit, pdf, param, label
    elif distribution == 'Rayleigh':
        # self-defined using polynomial fitting
        x = x_point ** 2
        y = np.log(1 - prob_cdf)
        p = np.polyfit(x, y, 1)
        sigma = (1 / (-2 * p[0])) ** 0.5
        pdf = x_point_fit / (sigma ** 2) * np.exp(
                -x_point_fit ** 2 / (2 * sigma ** 2))
        label = r'{name} ($\sigma$={sigma})'.format(name=distribution,
                                                    sigma=round(sigma, 2))
        return x_point_fit, pdf, sigma, label
    elif distribution == 'lognormal':
        # scipy.stats method
        s, loc, scale = stats.lognorm.fit(x_point, floc=0)
        mu = np.log(scale)
        sigma = s
        param = [mu, sigma]
        pi = math.pi
        pdf = 1 / x_point_fit * 1 / (sigma * (2 * pi) ** 0.5) * np.exp(
            -(np.log(x_point_fit) - mu) ** 2 / (2 * sigma ** 2))
        label = r'{name} ($\mu$={k}, $\sigma$={sigma})'.format(
                name=distribution, k=round(mu, 2), sigma=round(sigma, 2))
        return x_point_fit, pdf, param, label

""" This script provides methods to fit original data to several known 
distributions including Weibull, Rayleigh and log normal.

There are two ways to fitting.
1. If probability data is available, the parameters of distributions can be 
found by using polynomial fitting. This is a nice alternative to standard 
fitting functions.
2. If only raw/original data is available or method 1 is not applicable, using 
standard fitting functions provided in scipy.stats is the way.
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
        # Weibull distribution, self-defined using polynomial fitting
        x_wb = np.log(x_point)
        y_wb = np.log(-np.log(1 - prob_cdf))
        p_wb = np.polyfit(x_wb, y_wb, 1)
        k_wb = p_wb[0]
        lbd_wb = np.exp(-p_wb[1] / p_wb[0])
        param_wb = [k_wb, lbd_wb]
        pdf_wb = ((k_wb / lbd_wb) * (x_point_fit / lbd_wb) ** (k_wb - 1)
                  * np.exp(-(x_point_fit / lbd_wb) ** k_wb))
        # Weibull distribution using scipy.stats method
        param_sp_wb = stats.exponweib.fit(data_original)
        pdf_sp_wb = stats.exponweib.pdf(x_point_fit, *param_sp_wb)
        # choose return from the two methods and see which fits better
        param = param_sp_wb  # param_wb or param_sp_wb
        pdf = pdf_sp_wb  # pdf_wb or pdf_sp_wb
        label = r'{name} ($k$={k}, $\lambda$={lba})'.format(name=distribution, 
                  k=round(param[0], 2), lba=round(param[1], 2))
        return x_point_fit, pdf, param, label
    elif distribution == 'Rayleigh':
        # Rayleigh distribution, self-defined using polynomial fitting
        x_wb = np.log(x_point)
        x_ray = x_point ** 2
        y_ray = np.log(1 - prob_cdf)
        p_ray = np.polyfit(x_ray, y_ray, 1)
        sigma = (1 / (-2 * p_ray[0])) ** 0.5
        pdf_ray = x_point_fit / (sigma ** 2) * np.exp(
                -x_point_fit ** 2 / (2 * sigma ** 2))
        label = r'{name} ($\sigma$={sigma})'.format(
                name=distribution, sigma=round(sigma, 2))        
        return x_point_fit, pdf_ray, sigma, label
    elif distribution == 'lognormal':
        # log normal using standard scipy.stats function
        s, loc, scale = stats.lognorm.fit(x_point, floc=0)
        mu_ln = np.log(scale)
        sigma_ln = s
        param_ln = [mu_ln, sigma_ln]
        pi = math.pi
        pdf_ln = 1 / x_point_fit * 1 / (sigma_ln * (2 * pi) ** 0.5) * np.exp(
            -(np.log(x_point_fit) - mu_ln) ** 2 / (2 * sigma_ln ** 2))
        label = r'{name} ($\mu$={k}, $\sigma$={sigma})'.format(
                name=distribution, k=round(mu_ln, 2), sigma=round(sigma_ln, 2))        
        return x_point_fit, pdf_ln, param_ln, label

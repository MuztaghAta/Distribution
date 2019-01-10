"""This script provides self-defined methods to compute and plot CDF/PDF of a
series of data points and to fit the data to known statistical distributions.

Steps:
1. import data from file
2. compute CDF and PDF and save the results to file for future use
3. fit data to known distribution (using methods defined in fitting.py)
4. plot and save

You need to:
1. put the data file in the folder of this script and remove the header
2. change names in the 'import data' section according to your need
3. choose normal or log plot and plot formats in the 'plot and save' section
"""

import os.path
import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby
from fitting import fit_distribution  # self defined fitting methods


# step 1: import data
source_data_name = 'Douyin_video_size.csv'
dimension = 'size'  # bitrate, duration, size
unit = 'MB'  # Kpbs, seconds, MB
prob_data_name = 'probability_video_{}.txt'.format(dimension)
fig_name = 'video_{}.pdf'.format(dimension)
distribution2fit = 'Weibull'  # Weibull, Rayleigh, lognormal or leave it empty
# import data file
cwd = os.getcwd()
data_file = cwd + os.sep + source_data_name
data_whole = np.genfromtxt(data_file, delimiter=',')  # for csv file
# data_whole = np.loadtxt(data_file)  # for txt file
# remove title entry if there is
data_original = data_whole[~np.isnan(data_whole)]

# step 2: compute CDF/PDF or load if available
# check if the probability information has been available or not
prob_file = cwd + os.sep + prob_data_name
# read probability data from the file if it exists
if os.path.exists(prob_file):
    prob_all = np.loadtxt(prob_file)  # all probability data
    x_point = prob_all[0]  # x axis points (no duplicates)
    prob_pdf = prob_all[1]  # point probability
    prob_cdf = prob_all[2]  # cumulative probability
    num_duplicate = prob_all[3]  # frequency of x axis ticks
# compute the probability if not available and save it for future use
else:
    x_point = list(np.sort(data_original))  # sort the data in ascending order
    # find out how many duplicates each unique value or element in x_point has
    num_duplicate = [len(list(group)) for key, group in groupby(x_point)]
    # compute the cumulative probability of each unique data point
    prob_cdf = list(np.arange(1, len(x_point) + 1) / float(len(x_point)))
    # remove the first n-1 of n duplicates in x_point and prob_cdf so that one 
    # x point has only one probability
    x_point_rev = x_point[::-1]  # sort the x points in descending order
    prob_cdf_rev = prob_cdf[::-1]  # sort the probability in descending order
    seen = set()
    duplicate = []
    for idx, item in enumerate(x_point_rev):
        if item not in seen:
            seen.add(item)
        else:
            duplicate.append(idx)
    # remove the elements with indices in duplicate
    already_removed = 0
    for idx in duplicate:
        del x_point_rev[idx - already_removed]
        del prob_cdf_rev[idx - already_removed]
        already_removed += 1
    # compute point probability based on prob_cdf_rev and x_point_rev
    cdf_dif_rev = [x - y for x, y in zip(prob_cdf_rev, prob_cdf_rev[1:])]
    x_dif_rev = [x - y for x, y in zip(x_point_rev, x_point_rev[1:])]
    prob_pdf_rev = list(np.array(cdf_dif_rev)/np.array(x_dif_rev))
    # reverse back the data vector and probability in ascending order
    x_point = x_point_rev[::-1]
    prob_pdf = prob_pdf_rev[::-1]
    prob_cdf = prob_cdf_rev[::-1]
    # remove the first point of x_point, prob_cdf, num_duplicate,
    # because prob_pdf doesn't have this data point
    x_point = x_point[1:]
    prob_cdf = prob_cdf[1:]
    num_duplicate = num_duplicate[1:]
    # save the probability results
    prob_all = np.array([x_point, prob_pdf, prob_cdf, num_duplicate])
    np.savetxt(prob_data_name, prob_all)

# step 3: fit data to known distributions
if not distribution2fit == '':
    x_point_fit, prob_pdf_fit, param, label_dis = fit_distribution(
            prob_all, data_original, distribution2fit)

# step 4: plot and save
log_plot = 0  # 0 - normal plot, 1 - log plot
ms = 3  # marker size
lw = 2  # line width
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax2 = ax1.twinx()
if log_plot == 1:
    ax1.loglog(x_point, prob_pdf)
    ax2.loglog(x_point, prob_cdf)
else:
    ax1.stackplot(x_point, prob_pdf, color='red', alpha=0.4, 
                  labels=['PDF of the original data'])
    if not distribution2fit == '':
        ax1.plot(x_point_fit, prob_pdf_fit, color='red', linewidth=lw, 
                 markersize=ms, alpha=0.8, label=label_dis)
    ax2.plot(x_point, prob_cdf, color='C0', linewidth=lw, markersize=ms, 
             alpha=1.0, label='CDF of the original data')
ax1.set_ylabel('PDF', color='red')  # Number of videos, PDF
ax2.set_ylabel('CDF', color='C0')
ax1.set_xlabel('Video {} ({})'.format(dimension, unit))
fig.legend(loc='center right', bbox_to_anchor=(0.8, 0.5), frameon=False)
plt.show()
# save the plot, bbox_inches="tight" makes the plot to show axis labels in full
fig.savefig(fig_name, bbox_inches="tight")

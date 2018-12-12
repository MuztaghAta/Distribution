""" This script is to plot the PDF, CDF and fitting of a given data.

step 1: import data from file
step 2: prepare/process data for plotting and save them prepared data to file for future use
step 3: fit data to known distribution
step 4: plot using the prepared data and save figure

You need to:
1. put the data file in the folder of this script and remove the header
2. change source_data_name and prob_data_name
3. change the id_column according to your need
4. choose normal plot or log plot
5. change the format (marker, line, color, label, ...) of the plot
6. check line 52-53 to see whether you need to remove the largest elements
"""

import os.path
import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby
from fitting import fit_distribution  # self defined fitting function


# customization
dimension = 'bitrate'  # bitrate, duration, size
source_data_name = 'sample.csv'
prob_data_name = 'probability_video_{}.txt'.format(dimension)
fig_name = 'video_{}.pdf'.format(dimension)

distribution2fit = 'Weibull'  # choose from Weibull, Rayleigh, lognormal or leave it empty

cwd = os.getcwd()

data_file = cwd + os.sep + source_data_name
data_whole = np.genfromtxt(data_file, delimiter=',')  # for csv file
# data_whole = np.loadtxt(data_file)  # for txt file

# id_column = 0  # which column of data do you want to plot
# data_original = data_whole[:, id_column]
data_original = data_whole

# check if the probability information has been available or not
prob_file = cwd + os.sep + prob_data_name
# read probability data from the file if it exists
if os.path.exists(prob_file):
    prob_all = np.loadtxt(prob_file)  # all probability data
    x_point = prob_all[0]  # x axis ticks (no duplicates)
    prob_pdf = prob_all[1]  # point probability
    prob_cdf = prob_all[2]  # cumulative probability
    num_duplicate = prob_all[3]  # frequency of x axis ticks, similar with prob_pdf but with different scale
# if the probability is not available yet, compute from original data and save it for future use
else:
    # sort the data in an ascending order, it doesn't remove duplicates
    x_point = list(np.sort(data_original))
    # # remove the largest elements which are regarded as abnormal !!!
    # x_point.remove(max(x_point))
    # find out how many duplicates each unique number/element in x_point has
    num_duplicate = [len(list(group)) for key, group in groupby(x_point)]
    # compute the cumulative probability of each point
    prob_cdf = list(np.arange(1, len(x_point) + 1) / float(len(x_point)))

    # remove the first n-1 of n duplicates in x_point and prob_cdf so that one x tick has only one probability
    x_point_rev = x_point[::-1]  # sort the data vector in descending order
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

    # compute the probability of each point based on prob_cdf_rev and x_point_rev
    cdf_dif_rev = [x - y for x, y in zip(prob_cdf_rev, prob_cdf_rev[1:])]
    x_dif_rev = [x - y for x, y in zip(x_point_rev, x_point_rev[1:])]
    prob_pdf_rev = list(np.array(cdf_dif_rev)/np.array(x_dif_rev))
    # prob_rev.append(prob_cdf_rev[-1])
    # reverse back the data vector and probability in ascending order
    x_point = x_point_rev[::-1]
    prob_pdf = prob_pdf_rev[::-1]
    prob_cdf = prob_cdf_rev[::-1]
    # remove the first point of x_point, prob_cdf, num_duplicate,
    # because they have one more data point than prob_pdf, which is the smallest point
    x_point = x_point[1:]
    prob_cdf = prob_cdf[1:]
    num_duplicate = num_duplicate[1:]
    # save the probability results as a list
    prob_all = np.array([x_point, prob_pdf, prob_cdf, num_duplicate])
    np.savetxt(prob_data_name, prob_all)


# fit data to known distribution
if not distribution2fit == '':
    x_point_fit, prob_pdf_fit, param = fit_distribution(prob_all, data_original, distribution2fit)


# start to plot
log_plot = 0  # 0 - normal plot or 1 - log plot

# fmt_pdf = 'r.'  # plot format = '[color][marker][line]'
ms = 3  # marker size
lw = 2  # line width

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax2 = ax1.twinx()
if log_plot == 1:
    # ax1.loglog(x_point, num_duplicate, fmt_pdf)
    ax1.loglog(x_point, prob_pdf)
    ax2.loglog(x_point, prob_cdf)
else:
    ax1.stackplot(x_point, prob_pdf, color='red', alpha=0.4, labels=['PDF of the original data'])
    # ax1.plot(x_point, prob_pdf, color='red', markersize=ms, alpha=0.4, label='PDF of original data')
    if  not distribution2fit == '':
        label_dis = 'Fitted {} distribution'.format(distribution2fit)
        ax1.plot(x_point_fit, prob_pdf_fit, color='red', linewidth=lw, markersize=ms, alpha=0.8, label=label_dis)
    ax2.plot(x_point, prob_cdf, color='C0', linewidth=lw, markersize=ms, alpha=1.0, label='CDF of the original data')
    # colors: C0-C9, xkcd:azure ...
    # https://matplotlib.org/users/dflt_style_changes.html

# ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))  # scientific notation
ax1.set_ylabel('PDF', color='red')  # Number of videos, PDF
ax2.set_ylabel('CDF', color='C0')
ax1.set_xlabel('Video {}'.format(dimension))
fig.legend(loc='center right', bbox_to_anchor=(0.9, 0.5), frameon=False)
# legend location https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html

plt.show()
# save the plot, bbox_inches="tight" makes the plot to show axis labels in full
fig.savefig(fig_name, bbox_inches="tight")

# more formats: legend, ticks, x_lim, y_lim, font size, ...

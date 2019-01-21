"""
Zipf's law

This program fits data ranked along certain dimension (e.g. city population
and word appearance) to Zipfian distribution. The probability mass function
for zipf is:
pmf(k, a) = 1/(zeta(a) * k**a), for k >= 1.
"""
import os.path
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
import seaborn as sns


# import data
dimension = 'play'  # bitrate, duration, size
source_data_name = 'DouYinData_26wcsvcopy_{}.csv'.format(dimension)
prob_data_name = 'probability_video_{}.txt'.format(dimension)

cwd = os.getcwd()
data_file = cwd + os.sep + source_data_name
data_original = np.genfromtxt(data_file, delimiter=',')
data_whole = data_original[~np.isnan(data_original)]
data_unique = np.unique(data_whole)

# remove duplicates and rank the frequencies (sort them in descending order)
frequency = np.sort(data_unique)[::-1]
# truncate data if only part of the data is interested
# frequency = frequency[0:120000]
rank = np.arange(1, len(frequency)+1)
pmf = frequency / sum(frequency)


def zipf_fitting(rk, freq):
    """Zipf pmf(or normalized frequency) fitting with rank and frequency
    Maths: f(rk) = 1 / (zeta(a) * rk**a) => log(f(rk)) = - log(z(a)) - a*log(x)
    Use numpy.polyfit (or scipy.polyfit) to find a and then we get f(rk) easily
    """
    x = np.log(rk)
    y = np.log(freq / sum(freq))
    p = np.polyfit(x, y, 1)
    a = -p[0]
    return 1/(special.zeta(a) * rk**a), a


pmf_z, alpha = zipf_fitting(rank, frequency)
alpha = round(alpha, 2)  # keep the first two decimal

# plot fitting result
log_plot = 1  # 0 - normal plot, 1 - log plot
sns.set()
if log_plot == 1:
    # plt.loglog(rank, frequency)
    plt.loglog(rank, pmf)
    plt.loglog(rank, pmf_z)
else:
    # plt.plot(rank, frequency)
    plt.plot(rank, pmf)
    plt.plot(rank, pmf_z)
plt.xlabel('Ranking of videos in terms of number of views')
plt.ylabel('PMF')
lbs = ['Orignal data', 'Zipf distribution $\\alpha$={}'.format(alpha)]
plt.legend(lbs, loc='upper right', bbox_to_anchor=(1, 1), frameon=False)
plt.tight_layout()
plt.show()

"""
Zipf's law

This program fits data ranked along certain dimension (e.g. city population
and word appearance) to Zipfian distribution. The probability mass function
for zipf is: pmf(x, a) = 1/(zeta(a) * x**a), for x >= 1 and a > 1.
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.zipf.html
It's clear that fitting data to zipf is essentially find a.

HOWEVER, the above function fails to characterize zipf if a <= 1. Therefore,
we resort to more original maths expression:
f(x) = (1/x**a) / sum_1^N (1/x**a), where N is the number of elements.
https://en.wikipedia.org/wiki/Zipf%27s_law
The right most part: sum_1^N (1/x**a) ~ (N**(1-a)-1) / (1-a)
https://en.wikipedia.org/wiki/Euler%E2%80%93Maclaurin_formula
This step significantly reduce computational complexity.
"""
import os.path
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
import seaborn as sns


# import data
dimension = 'share'  # play, share, like, comment
source_data_name = 'DouYinData_26wcsvcopy_{}.csv'.format(dimension)
# source_data_name = 'example.csv'
cwd = os.getcwd()
data_file = cwd + os.sep + source_data_name
data_original = np.genfromtxt(data_file, delimiter=',')
data_whole = data_original[~np.isnan(data_original)]
data_unique = np.trim_zeros(np.unique(data_whole))
# remove duplicates and rank the frequencies (sort them in descending order)
frequency = np.sort(data_unique)[::-1]
# truncate data if only part of the data is interested
frequency = frequency[0:1000]
rank = np.arange(1, len(frequency)+1)
pmf = frequency / sum(frequency)

# Zipf pmf(or normalized frequency) fitting with rank and frequency
# Maths: f(x) = 1 / (c * x**a) => log(f(x)) = - log(c) - a*log(x)
# Use numpy.polyfit (or scipy.polyfit) to find a and then we get f(x) easily
x = np.log(rank)
y = np.log(frequency / sum(frequency))
p = np.polyfit(x, y, 1)
a = -p[0]
if a > 1:
    c1 = special.zeta(a)
    c2 = rank ** a
    pmf_z = 1 / (special.zeta(a) * rank ** a), a
else:
    n = len(frequency)
    pmf_z = (1-a) / ((n**(1-a) - 1) * rank ** a)

a = round(a, 3)  # keep the three two decimal

# plot fitting result
log_plot = 1  # 0 - normal plot, 1 - log plot
format_on = 1  # 0 off, 1 on
sns.set()
if log_plot == 1:
    plt.loglog(rank, pmf, 'o')
    plt.loglog(rank, pmf_z, 'red', linewidth=2)
else:
    plt.plot(rank, pmf, 'o')
    plt.plot(rank, pmf_z, 'red', linewidth=2)
if format_on == 1:
    plt.xlabel('Ranking of videos in terms of number of shares')
    plt.ylabel('PMF')
    lbs = ['Orignal data', 'Zipf distribution $\\alpha$={}'.format(a)]
    plt.legend(lbs, loc='upper right', bbox_to_anchor=(1, 1), frameon=False)
    plt.tight_layout()
plt.show()

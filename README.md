# Distribution
Given a series of data points (e.g. the sizes of many videos), we are interested in visualizing the data in terms of PDF and CDF, and fitting the curve to known distributions in order to charecterize the data in a scientific way and to get deep insights from the data. Here provides an example data about the size of many short videos from a popular platform. 

'distribution.py' and 'fitting.py' together can fit data to continous statistical distributions including Weibull, Rayleigh and lognormal. 'zipf.py' alone can fit data to the Zipfian distribution. Zipfian distribution is discrete and thus not integrated in the 'fitting.py' module.

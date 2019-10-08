import numpy as np
import matplotlib.pyplot as plt

"""
Question 2

Write a function that generates a specified number of independent and identically distributed
samples  paired  with  the  class  labels  that  generated  these  samples.   Specifically,  the  data
distribution  is  a  mixture  of  Gaussians  with  specified  prior  probabilities  for  each  Gaussian
class conditional pdf, as well as respective mean vectors and covariance matrices.  Generate
and visualize data in the form of scatter plots, with a color/marker based identification of
the class label for each sample for each of the following cases (using Matlab syntax for 22
matrices):

1.  Number  of  samples  =  400;  class  means  [0,0]T and  [3,3]T;  class  covariance  matrices
both set to I; equal class priors.
2.  All parameters same as (1), but both covariance matrices changed to [3,1; 1,0.8].
3.  Number  of  samples  =  400;  class  means  [0,0]T and  [2,2]T;  class  covariance  matrices
[2,0.5; 0.5,1] and [2,1.9; 1.9,5]; equal class priors.
4.  Same (1), but prior for class priors are 0.05 and 0.95.
5.  Same (2), but prior for class priors are 0.05 and 0.95.
6.  Same (3), but prior for class priors are 0.05 and 0.95.

Make sure your plots include axis labels, titles, and data legends. Describe how your sampling
procedure works, using zero-mean identity-covariance Gaussian sample generators.  Addition-
ally, for each of these datasets, use the maximum-a-priori (MAP) classification rule (using
full knowledge of the respective data pdfs) and produce inferred class labels for each data
samples.  In accompanying visualizations, demonstrate scatter plots of the data for each case
along with their inferred (decision) labels.  For each case,  count the number of errors and
estimate the probability of error based on these counts.
"""
mu = []
sigma = []
samps = 0

def gauss(mu, sigma, samps):
     x = (sigma**(1/2)) * np.random.randn(length(mu), samps) + np.matlib.repmat()
     return x

"""
Question 3

For the datasets you generated in Question 2, implement and apply the Fisher Linear Discriminant 
Analysis classifier with the decision threshold for the linear discriminant score set
to minimize the smallest probability error you can achieve on the specific data sets generated
for each case.  Visualize the one-dimensional Fisher LDA discriminant scores and decision
labels for each sample in separate plots for each case.  Note: We will soon discuss the principle
of cross-validation that dictates parameter selection and performance assessment must use
independent datasets.
"""
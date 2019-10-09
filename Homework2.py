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

def gaussClass(mu, sigma, samps, p, n):
    x0 = []
    y0 = []
    x1 = []
    y1 = []
    data = []
    for s in range(samps):
        c = np.random.rand(1)*100
        if c <= (p[0]*100):
            x = np.random.multivariate_normal(mu[0], sigma[0])
            x0.append(x[0])
            y0.append(x[1])
            data.append(x)
        else:
            x = np.random.multivariate_normal(mu[1], sigma[1])
            x1.append(x[0])
            y1.append(x[1])
            data.append(x)
    ax1, ax2 = plt.subplots(2)
    ax1.scatter(x0, y0, marker='o', color='blue')
    ax1.scatter(x1, y1, marker='x', color='red')
    ax1.xlabel('x1')
    ax1.ylabel('x2')
    ax1.title('Gaussian class labels for part ' + str(n))
    ax1.legend(['Class 0', 'Class 1'])
    plt.show()
    return

p0 = [0.5, 0.5]
p1 = [0.05, 0.95]

mu = [[0, 0], [3, 3]]
sigma = [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]
samps = 400
p = p0
gaussClass(mu, sigma, samps, p, 1)
p = p1
gaussClass(mu, sigma, samps, p, 4)

sigma = [[[3, 1], [1, 0.8]], [[3, 1], [1, 0.8]]]
p = p0
gaussClass(mu, sigma, samps, p, 2)
p = p1
gaussClass(mu, sigma, samps, p, 5)

mu = [[0, 0], [2, 2]]
sigma = [[[2, 0.5], [0.5, 1]], [[2, 1.9], [1.9, 5]]]
p = p0
gaussClass(mu, sigma, samps, p, 3)
p = p1
gaussClass(mu, sigma, samps, p, 6)

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
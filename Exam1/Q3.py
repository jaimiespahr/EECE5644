import numpy as np
from scipy.stats import norm, uniform
import matplotlib.pyplot as plt

sigma = .7
B = range(-5, 5)
wTrue = np.array([.5, -.2, 0, 1])

def wMAP(gamma):
    priorCov = np.dot(gamma ** 2, np.eye(4))
    prior = norm(0, priorCov)
    pXYgivenW =
    return

N = 10
gamma = [10**b for b in B]
v = norm(0, sigma**2)
xVec = [x**3, x**2, x, 1]
w = np.transpose([a, b, c, d])
y = xVec * w + v.rvs(np.shape(xVec)[0])
x = uniform(-1, 1).rvs()
import numpy as np
import matplotlib.pyplot as plt

"""
Generate a plot of the log-likelihood-ratio function for the case a_1=0, b_1=1
    and a_2=1, b_2=2. Label axes l(x) and x, include a title and caption.
"""

a = [0, 1]
b = [1, 2]

def llr(x):
    term1 = -np.abs(x-a[0]) / b[0]
    term2 = np.abs(x-a[1]) / b[1]
    l = term1 + term2
    return l

x = np.linspace(-10, 10, 500)
fig = plt.figure()
plt.title('Log-likelihood-ratio for Parametric pdfs')
plt.xlabel('x')
plt.ylabel('l(x)')
plt.plot(x, llr(x))
plt.show()
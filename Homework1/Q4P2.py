import numpy as np
import matplotlib.pyplot as plt

"""
Generate plots for the class-conditional pdfs p(x|L=l) for l in {1,2} and the
    class-posterior probabilities p(L=l|x) for l in {1,2}, and show the decision
    boundary for the case mu=1, var=2.
"""

mu1 = 0
mu2 = 1
var1 = 1
var2 = 2

x = np.linspace(-5, 5, 100)
arrayX = np.array(x)

fig, axes = plt.subplots(ncols=2)
axes[0].set_title('Class-conditional pdfs')
axes[1].set_title('Class posterior probabilities')
axes[0].set_xlabel('x')
axes[0].set_ylabel('p(x|L=l)')
axes[1].set_xlabel('x')
axes[1].set_ylabel('p(L=l|x)')

sig2 = np.sqrt(var2)
bound1 = -2.84
bound2 = 0.84

cond1 = (1/(np.sqrt(2*np.pi*var1))) * np.exp((-(arrayX-mu1)**2/(2*var1)))
cond2 = (1/(np.sqrt(2*np.pi*var2))) * np.exp((-(arrayX-mu2)**2/(2*var2)))
axes[0].plot(x, cond1, x, cond2)
axes[0].axvline(bound1)
axes[0].axvline(bound2)

prior = 1/2
post1 = prior*cond1
post2 = prior*cond2

axes[1].plot(x, post1, x, post2)
plt.show()
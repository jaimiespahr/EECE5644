import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

unitCircle = stats.multivariate_normal(mean=[0,0], cov=np.eye(2))
trueLoc = unitCircle.rvs(1)
sigNoise = .3
sigX = .25
sigY = .25
noise = stats.norm(0, sigNoise**2)

def prior(locs):
    p = []
    for x in locs:
        tempP = stats.multivariate_normal(mean=x, cov=[[sigX ** 2, 0], [0, sigY ** 2]]).pdf(x)
    p.append(tempP)
    return p

def conditional(x, R):
    p = stats.norm(np.mean(R), np.var(R)).pdf(x)
    return p

fig, axes = plt.subplots(2, 2)
axs = axes.flatten()
for K in [1, 2, 3, 4]:
    R = [-1]
    r = -np.infty
    refLocs = []
    while r in R < 0:
        refLocs = unitCircle.rvs(K)
        d = np.linalg.norm([refLocs - trueLoc])
        n = noise.rvs(K)
        R = d + n
    xAxis = np.linspace(-2, 2)
    yAxis = np.linspace(-2, 2)
    potLocs = []
    for i in xAxis:
        for j in yAxis:
            potLocs.append([i, j])
    z = np.log(conditional(potLocs, R) * prior(potLocs))
    axs[K-1] = plt.contour(potLocs, z, extent=[-2, 2, -2, 2])
    # plt.plot(trueLoc[0], trueLoc[1], marker='X')
    # for loc in refLocs:
    #     plt.plot(loc[0], loc[1], marker='O')
fig.show()
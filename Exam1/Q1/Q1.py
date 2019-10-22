import numpy as np
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt, lines as mlines, patches as mpatch


# Given in the problem
prior = [0.15, 0.35, 0.5]
mu = [np.array([-1, 0]), np.array([1, 0]), np.array([0, 1])]
sigma = [np.array([[1, -.4], [-.4, .5]]), np.array([[.5, 0], [0, .2]]), np.array([[.1, 0], [0, .1]])]
sampN = 10000

# Data generated from generateData_Exam1Question1.m
cTrue = np.loadtxt('LData.txt', delimiter=',')
locs = np.loadtxt('xData.txt', delimiter=',').transpose()

# Classify data
cGuess = []
for x in locs:
    p = []
    for i in range(3):
        pTemp = multivariate_normal(mean=mu[i], cov=sigma[i]).pdf(x) * prior[i]
        p.append(pTemp)
    c = np.argmax(p) + 1
    cGuess.append(c)

# Actual number of samples from each class
c1N = len([c for c in cTrue if c == 1])
c2N = len([c for c in cTrue if c == 2])
c3N = len([c for c in cTrue if c == 3])
print('Class 1: ' + str(c1N) + '\nClass 2: ' + str(c2N) + '\nClass 3: ' + str(c3N) + '\nTotal: ' + str(c1N+c2N+c3N))

# Confusion matrix
confused = np.zeros([3, 3])
for indx in range(len(cGuess)):
    i = int(cGuess[indx] - 1)
    j = int(cTrue[indx] - 1)
    confused[i, j] += 1
print(confused)

# Total number of samples misclassified
misClass = np.sum(confused) - np.diag(confused).sum()
print('Total number of misclassified samples: ' + str(int(misClass)))

# Estimated probability of error
pErr = misClass / sampN
print('Probability of error: ' + str(pErr))

# Visualization
fig = plt.figure()
marks = ['+', '.', 'x']
tColor = ['red', 'blue', 'yellow']
for indx in range(len(locs)):
    x = locs[indx][0]
    y = locs[indx][1]
    i = int(cGuess[indx] - 1)
    j = int(cTrue[indx] - 1)
    errColor = None
    if i == j:
        errColor = 'green'
    else:
        errColor = 'red'
    # plt.scatter(x, y, marker=marks[i-1], color=tColor[i-1])
    plt.scatter(x, y, marker=marks[j-1], color=errColor)
    if (indx % 50) == 0:
        print(indx)
plt.xlabel('x')
plt.ylabel('y')

# # For True Plot
# plt.title('True Class Labels')
# c1leg = mlines.Line2D([],[], color='red', marker='+', label='Class 1', linestyle='None')
# c2leg = mlines.Line2D([],[], color='blue', marker='.', label='Class 2', linestyle='None')
# c3leg = mlines.Line2D([],[], color='yellow', marker='x', label='Class 3', linestyle='None')
# plt.legend(handles=[c1leg, c2leg, c3leg])

# For Guess Plot
plt.title('Estimated Class Labels')
c1leg = mlines.Line2D([],[], marker='+', label='Class 1', linestyle='None')
c2leg = mlines.Line2D([],[], marker='.', label='Class 2', linestyle='None')
c3leg = mlines.Line2D([],[], marker='x', label='Class 3', linestyle='None')
gleg = mpatch.Patch(color='green', label='Correct')
rleg = mpatch.Patch(color='red', label='Incorrect')
plt.legend(handles=[c1leg, c2leg, c3leg, gleg, rleg])
fig.show()
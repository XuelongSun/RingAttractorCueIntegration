import numpy as np
from matplotlib import pyplot as plt

# parameters of simulation
T = 0.5
Ti = 0.001
dt = 1e-4
Nt = np.floor(T/dt)
Nti = np.floor(Ti/dt)
n = 100

# parameter cue 1
miu1 = 180
k1 = 40
sigma1s = np.linspace(5, 200, 40)

# parameter cue 2
miu2 = 90
k2 = 40
sigma2 = 40


def cue_integration(n, noise, sigma1):
    # ring attractor function
    Nn = n
    Dir = np.linspace(0, 360 - 360 / Nn, Nn).reshape(Nn, 1)

    # parameters of ring attractor network
    wEEk = 45.0 / Nn
    wEE = np.zeros((Nn, Nn))
    sigma = 120.0
    for i in range(0, Nn):
        for j in range(0, Nn):
            diff = np.min([np.abs(Dir[i] - Dir[j]), 360 - np.abs(Dir[i] - Dir[j])])
            wEE[i, j] = np.exp((-diff ** 2) / (2 * sigma ** 2))

    wEE = wEE * wEEk
    wIE = 60.0 / Nn

    wEI = -6.0
    wII = -1.0

    gammaE = -1.5
    gammaI = -7.5

    tauE = 0.005
    tauI = 0.00025

    # generate cue 1
    x1 = np.zeros((Nn, int(Nt)))
    diff = np.min([np.abs(Dir - miu1), 360 - np.abs(Dir - miu1)], axis=0)
    c1 = k1 * np.exp(-diff ** 2 / (2 * sigma1 ** 2)) / (np.sqrt(2 * np.pi) * sigma1)
    x1[:, int(Nti):] = np.repeat(c1.reshape(Nn, 1), int(Nt - Nti), axis=1)

    # generate cue 2
    x2 = np.zeros((Nn, int(Nt)))
    diff = np.min([np.abs(Dir - miu2), 360 - np.abs(Dir - miu2)], axis=0)
    c2 = k2 * np.exp(-diff ** 2 / (2 * sigma2 ** 2)) / (np.sqrt(2 * np.pi) * sigma2)
    x2[:, int(Nti):] = np.repeat(c2.reshape(Nn, 1), int(Nt - Nti), axis=1)

    # add noise
    x1 = x1 + noise * np.random.randn(Nn, int(Nt))
    x2 = x2 + noise * np.random.randn(Nn, int(Nt))

    # run integration
    c = np.zeros((Nn, int(Nt)))
    u = np.zeros((1, int(Nt)))
    c[:, 0] = 0.05 * np.ones(Nn, )

    for t in range(1, int(Nt)):
        c[:, t] = c[:, t - 1] + (-c[:, t - 1] + np.max(
            [np.zeros((Nn,)), gammaE + np.dot(wEE, c[:, t - 1]) + wEI * u[:, t - 1] + x1[:, t - 1]
             + x2[:, t - 1]], axis=0)) * dt / tauE
        u[:, t] = u[:, t - 1] + (-u[:, t - 1] + np.max([0, gammaI + wIE * np.sum(c[:, t - 1]) + wII * u[:, t - 1]],
                                                       axis=0)) * dt / tauI
    return x1[:, -1], x2[:, -1], c[:, -1]


d_RA = np.zeros(np.shape(sigma1s))
d_MLE = np.zeros(np.shape(sigma1s))
d_WTA = np.zeros((len(sigma1s)+1, 1))

for i in range(0,len(sigma1s)):
    x, y, z = cue_integration(n, 0, sigma1s[i])
    d_RA[i] = np.argmax(z)
    d_MLE[i] = (miu2 * sigma1s[i]**2 + miu1 * 40**2)/(sigma1s[i]**2 + 40**2)
    if sigma1s[i] <= 40:
        d_WTA[i] = miu1
        d_WTA[i+1] = miu1
    else:
        d_WTA[i+1] = miu2
        
d_RA = d_RA*360.0/n

# plot the result
fontsize = 30
fig1 = plt.figure(figsize=(19.2,10.8))
plt.plot(sigma1s, d_RA, label='RA', c='g')
plt.scatter(sigma1s, d_RA, marker='^', c='g')
plt.plot(sigma1s, d_MLE, 'b',label='MLE')
plt.scatter(sigma1s, d_MLE, marker='^', c='b')
x = np.append(np.linspace(5,40,8),np.linspace(40,200,33))
plt.plot(x[:8], d_WTA[:8], 'r',label='WTA')
plt.plot(x[9:], d_WTA[9:], 'r')
plt.plot([40,40,45],[180,90,90],'r')
plt.scatter(40,90,marker='^', c='R')
plt.scatter(x, d_WTA, marker='^', c='R')
plt.ylabel('Peak Position', fontdict={'size': fontsize, 'color': 'k'})
plt.ylim(60, 210)
plt.xlim(0, 210)
plt.plot([0, 210], [90, 90], c='k', ls='--')
plt.plot([0, 210], [180, 180], c='r', ls='--')
plt.text(5, 185, 'cue1 peak', fontdict={'size': fontsize, 'color': 'r'})
plt.text(5, 95, 'cue2 peak', fontdict={'size': fontsize, 'color': 'k'})
plt.legend(fontsize=fontsize)
plt.grid(1)
plt.xlabel(r'The Uncertainty of cue1 ($\sigma_{cue1}$)', fontdict={'size': fontsize, 'color': 'k'})
yticks = np.linspace(60, 210, int(n / 2))
plt.yticks(yticks, visible=0)

plt.show()
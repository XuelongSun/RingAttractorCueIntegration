import numpy as np
from matplotlib import pyplot as plt

# parameters of simulation
T = 0.5
Ti = 0.001
dt = 1e-4
Nt = np.floor(T/dt)
Nti = np.floor(Ti/dt)
n = 100

# parameter Path Integration
miu_PI = 110
k_PI = 40

# parameter Vision
miu_V = 0
k_V = 40
sigma_V = 70


def cue_integration(n, noise, sigma_PI):
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
    diff = np.min([np.abs(Dir - miu_PI), 360 - np.abs(Dir - miu_PI)], axis=0)
    c1 = k_PI * np.exp(-diff ** 2 / (2 * sigma_PI ** 2)) / (np.sqrt(2 * np.pi) * sigma_PI)
    x1[:, int(Nti):] = np.repeat(c1.reshape(Nn, 1), int(Nt - Nti), axis=1)

    # generate cue 2
    x2 = np.zeros((Nn, int(Nt)))
    diff = np.min([np.abs(Dir - miu_V), 360 - np.abs(Dir - miu_V)], axis=0)
    c2 = k_V * np.exp(-diff ** 2 / (2 * sigma_V ** 2)) / (np.sqrt(2 * np.pi) * sigma_V)
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


# home vector length
L = np.linspace(0.5,7,14)
L = np.hstack([0.001,L])
L_ = np.array([0.001,1,3,7])

# uncertainty of PI is changing as the increasing of home vector length
sigma_PIs = 1/L * sigma_V
pi = np.zeros([len(L),n])
v = np.zeros([len(L),n])
c = np.zeros([len(L),n])
dc = np.zeros([len(L),1])
for i in range(0,len(sigma_PIs)):
    pi[i,:],v[i,:],c[i,:] = cue_integration(n, 0, sigma_PIs[i])
    index = np.argmax(c[i,:])
    dc[i] = index*360.0/n

MLE = (110 * sigma_V**2)/(sigma_V**2 + sigma_PIs**2)

# get the output of the sampled point
dc_ = np.zeros([len(L_), 1])
dc_[0] = dc[0]
dc_[1] = dc[2]
dc_[2] = dc[6]
dc_[3] = dc[14]

# plot the results
fontsize = 28
fig = plt.figure(figsize=(19.2,10.8))

plt.plot(L, dc, label='Ring Attractor Integration (0.5m interval)', c='g',lw=2)
plt.scatter(L, dc,marker='^', c='g')

plt.scatter(L_, dc_, marker='o', s=80, c='grey')
plt.plot(L_, dc_, c='grey', label='Ring Attractor Integration (Sample 0m,1m,3m,7m)', lw=2.5)

plt.plot(L, MLE, label='Optimal Integration Based on Directional Uncertainty', c='r', lw=2)


plt.title(r'The Integrated Direction with the PI Vector Length Increasing ($N=100, \xi=0.0$)', fontdict={'size': fontsize, 'color': 'k'})

plt.ylabel('Direction', fontdict={'size': fontsize, 'color': 'k'})
plt.ylim(-150, 150)
plt.xlim(0, 8)
plt.plot([0, 8], [0, 0], c='k', ls='--')
plt.plot([0, 8], [110, 110], c='k', ls='--')
plt.text(5, 115, 'PI Direction', fontdict={'size': fontsize-5, 'color': 'k'})
plt.text(5, 5, 'Vision Direction', fontdict={'size': fontsize-5, 'color': 'k'})
plt.legend(fontsize=fontsize-5)
plt.grid(1)

# save the fig as PDF file
# plt.xlabel(r'PI Vector Length (m)', fontdict={'size': fontsize, 'color': 'k'})

plt.show()
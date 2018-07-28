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
miu_PI = 90
k_PIs = [40, 20, 5]
sigma_PI = 40

# parameter Vision
miu_V = 160
k_V = 40
sigma_V = 40


def cue_integration(n, noise, k_PI):
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


def plotter(n,pi,v,c,k_PI):
    Nn = n
    Dir = np.linspace(0, 360 - 360 / Nn, Nn).reshape(Nn, 1)
    fontsize_k = 30
    fig = plt.figure(figsize=(19.2, 10.8))
    plt.plot(Dir, pi, color='red', lw=2.0, label='PI Signal')
    plt.plot(Dir, v, color='black', lw=2.0, label='Vision Signal')
    plt.plot(Dir, c, color='green', lw=2.0, label='Ring Attractor Integration')
    # plt.plot(Dir, opt_it1, 'b--', lw=2.0, label='MLE')
    plt.title(r'Activation Profile for PI Descending to Zero($N=100, \xi=0.0$)',
              fontdict={'size': fontsize_k, 'color': 'k'})
    plt.text(0, 0.6, r'$K_{Vision}=40$', fontdict={'size': fontsize_k - 4, 'color': 'k'})
    plt.text(0, 0.55, r'$K_{PI}=$' + str(k_PI), fontdict={'size': fontsize_k - 4, 'color': 'r'})
    plt.legend(fontsize=fontsize_k - 4)
    plt.ylabel('Activation', fontdict={'size': fontsize_k, 'color': 'k'})
    xticks = np.linspace(0, 360, Nn)
    plt.xticks(xticks, visible=0)
    ymax = np.max(c)
    plt.ylim(-0.05, ymax + 0.05)
    # plt.xlabel('Neurons Labelled with Preferences', fontdict={'size': fontsize_k, 'color': 'k'})
    plt.grid(1)
    return fig


# sub_fig 1-3
x, y, z = cue_integration(n, 0, k_PIs[0])
sub_fig1 = plotter(n,x,y,z,k_PIs[0])

x, y, z = cue_integration(n, 0, k_PIs[1])
sub_fig2 = plotter(n,x,y,z,k_PIs[1])

x, y, z = cue_integration(n, 0, k_PIs[2])
sub_fig3 = plotter(n,x,y,z,k_PIs[2])



plt.show()
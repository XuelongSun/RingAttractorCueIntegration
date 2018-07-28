import numpy as np
from matplotlib import pyplot as plt

# parameters of simulation
T = 0.5
Ti = 0.001
dt = 1e-4
Nt = np.floor(T/dt)
Nti = np.floor(Ti/dt)

# parameter cue 1
miu1 = 90
k1 = 40
sigma1 = 40

# parameter cue 2
miu2 = 155
k2 = 40
sigma2 = 35


def cue_integration(n, noise):
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

    # MLE-optimal integration
    opt_sigma1 = np.sqrt(sigma1 ** 2 * sigma2 ** 2 / (sigma1 ** 2 + sigma2 ** 2))
    opt_d = (miu1 * sigma2 ** 2 + miu2 * sigma1 ** 2) / (sigma1 ** 2 + sigma2 ** 2)
    diff = np.min([np.abs(Dir - opt_d), 360 - np.abs(Dir - opt_d)], axis=0)
    opt_it = k1 * np.exp(-diff ** 2 / (2 * opt_sigma1 ** 2)) / (np.sqrt(2 * np.pi) * opt_sigma1)

    return x1[:, -1], x2[:, -1], c[:, -1], opt_it


def plotter(x1, x2, c, MLE, n, noise):
    # plot the result
    Nn = n
    Dir = np.linspace(0, 360 - 360 / Nn, Nn).reshape(Nn, 1)

    fontsize_k = 30
    fig = plt.figure(figsize=(19.2,10.8))
    plt.plot(Dir, x1, color='red', lw=2.0, label='cue1')
    plt.plot(Dir, x2, color='black', lw=2.0, label='cue2')
    plt.plot(Dir, c, color='green', lw=2.0, label='Integration')
    plt.plot(Dir, MLE, 'b--', lw=2.0, label='MLE')
    plt.title(r'Activation Profile ($N=$' + str(Nn) + r', $\xi=$' + str(noise)+ ')', fontdict={'size': fontsize_k, 'color': 'k'})
    plt.text(0, 0.6, r'$\sigma_{cue1}=$' + str(sigma1), fontdict={'size': fontsize_k, 'color': 'r'})
    plt.text(0, 0.55, r'$\sigma_{cue2}=$' + str(sigma2), fontdict={'size': fontsize_k, 'color': 'k'})
    plt.legend(fontsize=fontsize_k)
    plt.ylabel('Activation', fontdict={'size': fontsize_k, 'color': 'k'})
    xticks = np.linspace(0, 360, Nn)
    plt.xticks(xticks, visible=0)
    ymax = np.max(c)
    plt.ylim(-0.05, ymax + 0.05)
    plt.xlabel('Neurons Labelled with Preferences', fontdict={'size': fontsize_k, 'color': 'k'})
    plt.grid(1)

    return fig


# profile_1 N=100, noise=0
p1_x1, p1_x2, p1_c, p1_opt = cue_integration(100, 0.0)
p1_fig = plotter(p1_x1, p1_x2, p1_c, p1_opt, 100, 0.0)

# profile_2 N=100, noise=0.01
p2_x1, p2_x2, p2_c, p2_opt = cue_integration(100, 0.01)
p2_fig = plotter(p2_x1, p2_x2, p2_c, p2_opt, 100, 0.01)

# profile_3 N=8, noise=0
p3_x1, p3_x2, p3_c, p3_opt = cue_integration(8, 0.0)
p3_fig = plotter(p3_x1, p3_x2, p3_c, p3_opt, 8, 0.0)

plt.show()
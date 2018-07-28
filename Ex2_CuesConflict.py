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
miu1 = 0
k1 = 40
sigma1 = 40

# parameter cue 2
miu2s = np.linspace(0, 180, 37)
k2 = 40
sigma2s = [40, 35, 20]


def cue_integration(n, noise, miu2, sigma2):
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


def plotter(i):
    # plot the results

    fig = plt.figure(figsize=(19.2, 10.8))
    fontsize = 28
    plt.plot(miu2s, RA[i, :], 'g', lw=2.0, label='RA')
    plt.plot(miu2s, MLE[i, :], 'b', lw=2.0, label='MLE')
    plt.plot(miu2s, WTA[i, :], 'r', lw=2.0, label='WTA')
    if i == 0:
        WTA_ = np.repeat(miu1, len(miu2s))
        plt.plot(miu2s, WTA_, 'r', lw=2.0)
    plt.text(0, 130, r'$\sigma_{cue1}=$' + str(sigma1) + r'$, D_{cue1}=0$', fontdict={'size': fontsize, 'color': 'r'})
    plt.text(0, 120, r'$\sigma_{cue2}=$' + str(sigma2s[i]) + r'$, D_{cue2}=0-180$',
             fontdict={'size': fontsize, 'color': 'k'})
    plt.xlabel('Conflict between cues (degrees)', fontdict={'size': fontsize, 'color': 'k'})
    plt.ylabel('Peak Position', fontdict={'size': fontsize, 'color': 'k'})
    ax = plt.gca()
    ax.set_aspect(1)
    ticks = np.linspace(0, 190, 20)
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.grid(1)
    plt.legend(fontsize=fontsize - 2)
    return fig


def bio_data_plotter(x1, y1, x2, y2):
    # plot the biological data from Jeffery
    fontsize = 26
    color = 'whitesmoke'
    fig = plt.figure(facecolor=color, figsize=(19.2, 10.8))
    ax = plt.gca()
    ax.patch.set_facecolor(color)
    plt.plot(x1, y1, 'orange', lw=2.0, label='Re-weighting model')
    plt.plot(x2, y2, 'k', lw=2.0, label='Biological experiment')
    x = np.linspace(0, 180, 19)
    plt.plot(x, x, 'r', ls='-', lw=2.0, label='WTA-visual stimulus dominate')
    plt.legend(fontsize=fontsize - 4, facecolor=color)
    plt.scatter(x1, re_we, color='orange')
    plt.scatter(x2, bio_data, color='k')

    plt.text(-5, 120, r'cue1 is the original HD firing', fontdict={'size': fontsize, 'color': 'r'})
    plt.text(-5, 110, r'cue2 is the rotating light', fontdict={'size': fontsize, 'color': 'k'})
    # plt.text(0, 100, r'the firing packet rotates from 0deg ', fontdict={'size': 18, 'color': 'k'})
    plt.text(0, 102, '        (visual stimulus)', fontdict={'size': fontsize, 'color': 'k'})
    plt.xlabel('Conflict between cues (degrees)', fontdict={'size': fontsize, 'color': 'k'})
    plt.ylabel('Firing Packet Rotation (degrees)', fontdict={'size': fontsize, 'color': 'k'})
    ax = plt.gca()
    ax.set_aspect(1)
    ticks = np.linspace(0, 190, 20)
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.grid(1)
    return fig


MLE = np.zeros([len(sigma2s), len(miu2s)])
WTA = np.zeros([len(sigma2s), len(miu2s)])
RA = np.zeros([len(sigma2s), len(miu2s)])

# calculate the winner-take-all (WTA)
for i in range(0, len(sigma2s)):
    WTA[i, :] = miu2s

# calculate optimal integration (MLE)
for i in range(0, len(sigma2s)):
    MLE[i, :] = (miu1 * sigma2s[i]**2 + miu2s * sigma1**2)/(sigma1**2 + sigma2s[i]**2)

# the output of ring attractor (RA)
for i in range(0, len(miu2s)):
    for j in range(0,len(sigma2s)):
            x, y, z = cue_integration(n, 0, miu2s[i], sigma2s[j])
            RA[j, i] = np.argmax(z)*360.0/n


# the re-weighting model and the biological data
re_we = np.array([7.701355,15.586731,23.721359,32.037277,40.442535,48.879074,57.369461,65.95433,74.497665,82.71228,
                  89.957062,93.534973,74.852859,46.019714,29.101624,16.806488,7.618546,-0.00061])
bio_data = np.array([13.7,21.2,41.2,64.5,66,93.7,69.8,50.9,29.6])

# fig1-3
fig1 = plotter(0)
fig2 = plotter(1)
fig3 = plotter(2)

# fig4
x1 = np.linspace(10,180,18)
x2 = np.linspace(20,180,9)
fig4 = bio_data_plotter(x1, re_we, x2, bio_data)

plt.show()
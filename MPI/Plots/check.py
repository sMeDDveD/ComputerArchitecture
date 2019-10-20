import matplotlib.pyplot as plt
import numpy as np
with plt.xkcd():
    # Based on "Stove Ownership" from XKCD by Randall Munroe
    # https://xkcd.com/418/

    fig = plt.figure()
    ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim([-20, 10])

    data = np.ones(100)
    data[70:] += np.arange(30)

    ax.annotate(
        'ШАС СКАЗАЛ,\n ЧТО Я МОЛОДЕЦ',
        xy=(70, 1), arrowprops=dict(arrowstyle='->'), xytext=(30, -10))

    ax.plot(data)

    ax.set_xlabel('время')
    ax.set_ylabel('моя самооценка')
    fig.text(
        0.5, 0.05,
        'Deem Solobuher',
        ha='center')
    plt.show()
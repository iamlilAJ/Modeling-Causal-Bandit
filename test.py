import yaml
from run_experiment import run_exp
from matplotlib import pyplot as plt
import numpy as np
np.random.seed(42)

def plot_comparison(a, b, c):
    # Convert to NumPy arrays
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    # Calculate mean and standard deviation for a and b
    a_mean = a.mean(axis=0)
    a_std = a.std(axis=0)
    b_mean = b.mean(axis=0)
    b_std = b.std(axis=0)
    c_mean = c.mean(axis=0)
    c_std = c.std(axis=0)


    # Plot mean lines
    plt.plot(a_mean, label='baseline', color = 'green')
    plt.text(len(a_mean) - 1, a_mean[-1], f'{a_mean[-1]:.1f}', verticalalignment='bottom', horizontalalignment='right')
    # Shade areas for standard deviations
    plt.fill_between(range(len(a_mean)), a_mean - a_std, a_mean + a_std, alpha=0.2)



    if len(b[0]) != 0 :
        plt.plot(b_mean, label='model-based LinUCB', color = 'blue')
        plt.text(len(b_mean) - 1, b_mean[-1], f'{b_mean[-1]:.1f}', verticalalignment='bottom', horizontalalignment='right')
        plt.fill_between(range(len(b_mean)), b_mean - b_std, b_mean + b_std, alpha=0.2)


    if len(c[0]) != 0 :
        plt.plot(c_mean, label='online model-based LinUCB', color = 'yellow')
        plt.text(len(c_mean) - 1, c_mean[-1], f'{c_mean[-1]:.1f}', verticalalignment='bottom', horizontalalignment='right')
        plt.fill_between(range(len(c_mean)), c_mean - c_std, c_mean + c_std, alpha=0.2)

    plt.xlabel('round t')
    plt.ylabel('cummulative regret')
    plt.legend()
    plt.show()


with open('config.yaml', 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

a, b, c = run_exp(config)

plot_comparison(a,b,c)
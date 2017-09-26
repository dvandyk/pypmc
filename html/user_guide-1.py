import numpy as np
from pypmc.mix_adapt.variational import GaussianInference
from pypmc.tools import plot_mixture
import matplotlib.pyplot as plt

# data points
N = 500
data = np.random.normal(size=2*N).reshape(N, 2)
# maximum number of components in mixture
K = 6
vb = GaussianInference(data, components=K,
                       alpha=10*np.ones(K),
                       nu=3*np.ones(K))

# plot data and initial guess
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], color='gray')
initial_mix = vb.make_mixture()
plot_mixture(initial_mix, cmap='gist_rainbow')
x_range = (-4, 4)
y_range = x_range
plt.xlim(x_range)
plt.ylim(y_range)
plt.gca().set_aspect('equal')
plt.title('Initial')

# compute variational Bayes posterior
vb.run(prune=0.5*len(data) / K, verbose=True)

# obtain most probable mixture and plot it
mix = vb.make_mixture()
plt.subplot(1, 2, 2)
plt.scatter(data[:, 0], data[:, 1], color='gray')
plt.xlim(x_range)
plt.ylim(y_range)
plt.gca().set_aspect('equal')
plot_mixture(mix, cmap='gist_rainbow')
plt.title('Final')
plt.show()
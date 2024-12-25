#Hyperbolic
#https://docs.scipy.org/doc/scipy/tutorial/stats/continuous_genhyperbolic.html
#https://lars76.github.io/2020/07/24/implementing-poincare-embedding.html
# Hyperboloid geometry: https://imgur.com/w2wfk7V
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

p, a, b, loc, scale = 1, 1, 0, 0, 1
x = np.linspace(-10, 10, 100)

# plot GH for different values of p
plt.figure(0)
plt.title("Generalized Hyperbolic | -10 < p < 10")
plt.plot(x, stats.genhyperbolic.pdf(x, p, a, b, loc, scale),
        label = 'GH(p=1, a=1, b=0, loc=0, scale=1)')
plt.plot(x, stats.genhyperbolic.pdf(x, p, a, b, loc, scale),
        color = 'red', alpha = 0.5, label='GH(p>1, a=1, b=0, loc=0, scale=1)')
[plt.plot(x, stats.genhyperbolic.pdf(x, p, a, b, loc, scale),
        color = 'red', alpha = 0.1) for p in np.linspace(1, 10, 10)]
plt.plot(x, stats.genhyperbolic.pdf(x, p, a, b, loc, scale),
        color = 'blue', alpha = 0.5, label='GH(p<1, a=1, b=0, loc=0, scale=1)')
[plt.plot(x, stats.genhyperbolic.pdf(x, p, a, b, loc, scale),
        color = 'blue', alpha = 0.1) for p in np.linspace(-10, 1, 10)]
plt.plot(x, stats.norm.pdf(x, loc, scale), label = 'N(loc=0, scale=1)')
plt.plot(x, stats.laplace.pdf(x, loc, scale), label = 'Laplace(loc=0, scale=1)')
plt.plot(x, stats.pareto.pdf(x+1, 1, loc, scale), label = 'Pareto(a=1, loc=0, scale=1)')
plt.ylim(1e-15, 1e2)
plt.yscale('log')
plt.legend(bbox_to_anchor=(1.1, 1))
plt.subplots_adjust(right=0.5)

# plot GH for different values of a
plt.figure(1)
plt.title("Generalized Hyperbolic | 0 < a < 10")
plt.plot(x, stats.genhyperbolic.pdf(x, p, a, b, loc, scale),
        label = 'GH(p=1, a=1, b=0, loc=0, scale=1)')
plt.plot(x, stats.genhyperbolic.pdf(x, p, a, b, loc, scale),
        color = 'blue', alpha = 0.5, label='GH(p=1, a>1, b=0, loc=0, scale=1)')
[plt.plot(x, stats.genhyperbolic.pdf(x, p, a, b, loc, scale),
        color = 'blue', alpha = 0.1) for a in np.linspace(1, 10, 10)]
plt.plot(x, stats.genhyperbolic.pdf(x, p, a, b, loc, scale),
        color = 'red', alpha = 0.5, label='GH(p=1, 0<a<1, b=0, loc=0, scale=1)')
[plt.plot(x, stats.genhyperbolic.pdf(x, p, a, b, loc, scale),
        color = 'red', alpha = 0.1) for a in np.linspace(0, 1, 10)]
plt.plot(x, stats.norm.pdf(x, loc, scale),  label = 'N(loc=0, scale=1)')
plt.plot(x, stats.laplace.pdf(x, loc, scale), label = 'Laplace(loc=0, scale=1)')
plt.plot(x, stats.pareto.pdf(x+1, 1, loc, scale), label = 'Pareto(a=1, loc=0, scale=1)')
plt.ylim(1e-15, 1e2)
plt.yscale('log')
plt.legend(bbox_to_anchor=(1.1, 1))
plt.subplots_adjust(right=0.5)

plt.show()
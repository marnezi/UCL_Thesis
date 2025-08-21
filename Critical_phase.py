import numpy as np
import matplotlib.pyplot as plt

#define the number of isolated numbers and edge p 
n = 1000
p = np.linspace(0, 0.01, 1000)  # edge probability range

# critical phase
exact = n * (1 - p)**(n - 1)
approx = n * np.exp(-p * (n - 1))  
p_thresh = np.log(n) / n

# plot settings
fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(p, exact, label=r'Exact: $n(1-p)^{n-1}$', color='black', linewidth=2)
ax.plot(p, approx, '--', label=r'Approx: $n e^{-p(n-1)}$', color='orange', linewidth=2)

ax.axvline(p_thresh, linestyle=':', color='firebrick', linewidth=2,label=r'Threshold: $p=\frac{\log n}{n}$')
ax.set_title(r'Expected Isolated Vertices in Erdős–Rényi $G(n,p)$, $n=1000$')

ax.set_xlabel(r'Edge probability $p$')
ax.set_ylabel(r'Expected number of isolated vertices')

ax.legend(loc='upper right', frameon=True)
ax.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()

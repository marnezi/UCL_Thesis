import matplotlib.pyplot as plt
import numpy as np

# value of eta
etas = [np.inf, 50, 1, 0.5, 0]

# assign custom codes 
colors = {
    np.inf: "black",          
    50: "#A4BED5FF",           
    1: "red",           
    0.5: "#72874EFF",          
    0: "#476F84FF"              
}
   
# defining plot settings
x = np.linspace(-5, 5, 400)
leb_density = np.ones_like(x)

plt.figure(figsize=(10,6))

plt.plot(
    x, leb_density, linestyle="--", color=colors[np.inf],
    linewidth=2.5, label=r"Lebesgue ($\eta \to \infty$)"
)

for eta in etas:
    if eta == 0:
        spike = np.zeros_like(x)
        spike[np.argmin(np.abs(x))] = 1
        plt.plot(
            x, spike, linewidth=2.5, color=colors[0],
            label=r"Gaussian ($\eta=0$)"
        )
    elif np.isinf(eta):
        continue
    else:
        gaussian_density = np.exp(-x**2 / (2*eta)) / np.sqrt(2*np.pi*eta)
        plt.plot(
            x, gaussian_density, linewidth=2.5, color=colors[eta],
            label=fr"Gaussian ($\eta={eta}$)"
        )

plt.title("Effect of $\eta$ parameter on Gaussian weighting vs Lebesgue measure", fontsize=18)
plt.xlabel("x", fontsize=15)
plt.ylabel("Weight", fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylim(0, 1.1)
plt.legend(fontsize=13)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


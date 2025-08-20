import numpy as np
import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10,4))

# ℓ1 
t = np.linspace(0, 2*np.pi, 400)
c, s = np.cos(t), np.sin(t)
r = (np.abs(c)**1 + np.abs(s)**1)**(-1/1)
x1, y1 = r*c, r*s
ax1.plot(x1, y1, color="#3776ab", lw=2)
ax1.axhline(0, color="black"); ax1.axvline(0, color="black")
ax1.set_aspect("equal"); ax1.set_xlim(-1.2,1.2); ax1.set_ylim(-1.2,1.2)
ax1.set_xticks([]); ax1.set_yticks([])
ax1.set_title(r"$p=1$", fontsize=16)
ax1.text(0, -1.5, r"$\|x\|_{1}=\sum |x_i|$", ha="center", fontsize=16)

# ℓ2
r = (np.abs(c)**2 + np.abs(s)**2)**(-1/2)
x2, y2 = r*c, r*s
ax2.plot(x2, y2, color="#3776ab", lw=2)
ax2.axhline(0, color="black"); ax2.axvline(0, color="black")
ax2.set_aspect("equal"); ax2.set_xlim(-1.2,1.2); ax2.set_ylim(-1.2,1.2)
ax2.set_xticks([]); ax2.set_yticks([])
ax2.set_title(r"$p=2$", fontsize=16)
ax2.text(0, -1.5, r"$\|x\|_{2}=(\sum x_i^{2})^{1/2}$", ha="center", fontsize=16)

# ℓ inf
x3 = [-1,1,1,-1,-1]; y3 = [-1,-1,1,1,-1]
ax3.plot(x3, y3, color="#3776ab", lw=2)
ax3.axhline(0, color="black"); ax3.axvline(0, color="black")
ax3.set_aspect("equal"); ax3.set_xlim(-1.2,1.2); ax3.set_ylim(-1.2,1.2)
ax3.set_xticks([]); ax3.set_yticks([])
ax3.set_title(r"$p=\infty$", fontsize=16)
ax3.text(0, -1.5, r"$\|x\|_{\infty}=\max |x_i|$", ha="center", fontsize=16)

plt.suptitle(r"$L_p$ norms", fontsize=20, y=1.02)
plt.tight_layout()
plt.show()
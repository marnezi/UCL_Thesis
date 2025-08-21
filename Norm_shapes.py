import numpy as np
import matplotlib.pyplot as plt

fig, (ax0, ax1, ax2,ax3) = plt.subplots(1, 4, figsize=(14, 4))

def clean_axes(ax):
    ax.set_aspect("equal")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.6, 1.2)  
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    # axis
    ax.annotate("", xy=(1.2, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="black"))
    ax.annotate("", xy=(-1.2, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="black"))
    ax.annotate("", xy=(0, 1.2), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="black"))
    ax.annotate("", xy=(0, -1.6), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="black"))

#l 0.5
t = np.linspace(0, 2*np.pi, 400)
c, s = np.cos(t), np.sin(t)
r = (np.abs(c)**0.5 + np.abs(s)**0.5)**(-1/0.5)
x0, y0 = r*c, r*s
ax0.plot(x0, y0, color="#3776ab", lw=2)
clean_axes(ax0)
ax0.set_title(r"$p=0.5$", fontsize=16, pad=6)
ax0.text(1, -1.45, r"$\|x\|_{0.5}=(\sum {\sqrt{|x_i|}})^2$", ha="center", fontsize=14)


#l 1 
t = np.linspace(0, 2*np.pi, 400)
c, s = np.cos(t), np.sin(t)
r = (np.abs(c)**1 + np.abs(s)**1)**(-1/1)
x1, y1 = r*c, r*s
ax1.plot(x1, y1, color="#3776ab", lw=2)
clean_axes(ax1)
ax1.set_title(r"$p=1$", fontsize=16, pad=6)
ax1.text(0.8, -1.45, r"$\|x\|_{1}=\sum |x_i|$", ha="center", fontsize=14)

#l 2 
r = (np.abs(c)**2 + np.abs(s)**2)**(-1/2)
x2, y2 = r*c, r*s
ax2.plot(x2, y2, color="#3776ab", lw=2)
clean_axes(ax2)
ax2.set_title(r"$p=2$", fontsize=16, pad=6)
ax2.text(0.8, -1.45, r"$\|x\|_{2}=\sqrt{\sum x_i^{2}}$", ha="center", fontsize=14)

#l inf
x3 = np.array([-1, 1, 1, -1, -1])
y3 = np.array([-1,-1, 1,  1, -1])
ax3.plot(x3, y3, color="#3776ab", lw=2)
clean_axes(ax3)
ax3.set_title(r"$p=\infty$", fontsize=16, pad=6)
ax3.text(0.8, -1.45, r"$\|x\|_{\infty}=\max |x_i|$", ha="center", fontsize=14)

plt.suptitle(r"$L_p$ norms", fontsize=20, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


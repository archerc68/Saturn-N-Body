import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Animation parameters
FPS = 60
downsample = int(1e0)  # Downsample points recorded to points plotted

# Loading positions from directory
rs = np.load("Saturn/rsSaturn.npy")
R = np.load("Saturn/RSaturn.npy")


# Downsampling
rsdown = rs[::downsample]
Rdown = R[::downsample]


# Initialising plot
fig, ax = plt.subplots(1, 1)

# Astroid positions (even density in caertesian space)
r_max = 140.220e6  # Extent of Saturn's rings
r_min = 74.500e6  # Diameter of Saturn

ax.set_xlim(-1.5 * r_max, 1.5 * r_max)
ax.set_ylim(-1.5 * r_max, 1.5 * r_max)
ax.set_aspect("equal")

(particles,) = ax.plot([], [], "bo", ms=6, alpha=0.25)
(heavy,) = ax.plot([], [], "ro", ms=6)


def init():
    particles.set_data([], [])
    heavy.set_data([], [])
    return particles, heavy


# Animating
def animate(i):
    particles.set_data(rsdown[i, :, 0], rsdown[i, :, 1])
    heavy.set_data(Rdown[i, :, 0], Rdown[i, :, 1])
    return particles, heavy


# Plotting

anim = FuncAnimation(
    fig,
    animate,
    interval=1000 / FPS,
    frames=int(rs.shape[0] / downsample),
    init_func=init,
    blit=True,
)


# Saving .mp4 to directory
anim.save("Saturn/Saturn.mp4", dpi=250)

print("Plotted")

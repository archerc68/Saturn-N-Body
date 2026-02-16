import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Animation parameters
FPS = 60
downsample = int(1e1)  # Downsample points recorded to points plotted

# Loading positional data
rs = np.load("Earth-Moon/rs.npy")
R = np.load("Earth-Moon/R.npy")

R0 = R[0, 0, 0] + R[0, 1, 0]

# Downsampling
rsdown = rs[::downsample]
Rdown = R[::downsample]


# Initialising plot
fig, ax = plt.subplots(1, 1)

ax.set_xlim(-3 * R0, 3 * R0)
ax.set_ylim(-3 * R0, 3 * R0)
ax.set_aspect("equal")


# Initialasing particles
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
)


anim.save("Earth-Moon/Orbits.mp4", dpi=250)

print("Plotted")

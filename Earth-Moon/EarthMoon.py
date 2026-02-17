import numpy as np
from tqdm import tqdm

# Simulation parameters
dt, steps = 100, int(5e5)
downsample = 10  # Downsample points calculated to points recorded
N = 1000

print("Elapsed simulated time: " + str(steps * dt / 31556952) + " years")

# Massive bodies
M = np.array([6.1e24, 7.3e22])
d = 384.4e6
R0 = np.array([-M[1] / np.sum(M) * d, M[0] / np.sum(M) * d])
omega = np.sqrt(6.67e-11 * M[0] / R0[1] ** 3)


# Light bodies/particles
r_min, r_max = 0.01 * R0[1], 1.3 * R0[1]

theta = 2 * np.pi * np.random.random(N)


noise = r_min + (r_max - r_min) * np.random.random(N)[:, None]  # Radial profile


omega_r = np.sqrt(6.67e-11 * M[0] / noise**3)  # K3L for particles


# Particles' initial positions
r = np.ones((N, 2)) * noise
r[:, 0] *= np.cos(theta)
r[:, 1] *= np.sin(theta)


# Particles' initial velocities
v = np.ones((N, 2)) * omega_r * noise
v[:, 0] *= -np.sin(theta)
v[:, 1] *= np.cos(theta)


# Acceleration of light bodies
def acc(r, R):
    diff = R[None, :, :] - r[:, None, :]
    dx, dy = diff[..., 0], diff[..., 1]
    dist = np.sqrt(dx * dx + dy * dy + 1e8)     # Softened gravity
    isl = diff / (dist[..., None] ** 3)
    a = 6.67e-11 * np.einsum("ijk, j -> ik", isl, M)
    return a


# r += dr ()
def Kahan(r, dr, c):
    y = dr - c
    t = r + y
    c = (t - r) - y
    r = t.copy()
    return r, c


# Numerical integrator
def Verlet(r, v):
    # Setting up Kahan
    cr = np.zeros_like(r)
    cv = np.zeros_like(v)

    # History
    rs = np.empty((int(steps / downsample), N, 2))
    rs[0] = r.copy()

    R = np.empty((int(steps / downsample), 2, 2))
    R[0, :, 0] = R0

    # Substep initialisation
    R_sub = np.empty((downsample, 2, 2))
    dt_sub = np.arange(downsample) * dt

    for i in tqdm(range(1, int(steps / downsample))):
        # Heavy positions
        t_sub = dt_sub + (i - 1) * dt * downsample
        R_sub[..., 0] = R0[None, :] * np.cos(omega * t_sub[:, None])
        R_sub[..., 1] = R0[None, :] * np.sin(omega * t_sub[:, None])

        # Light positions
        for j in range(downsample):
            a0 = acc(r, R_sub[j])
            dr = v * dt + 0.5 * a0 * dt * dt
            r, cr = Kahan(r, dr, cr)

            a1 = acc(r, R_sub[j])
            dv = 0.5 * (a0 + a1) * dt
            v, cv = Kahan(v, dv, cv)

        # Storing positions
        rs[i] = r.copy()
        R[i] = R_sub[-1].copy()

    print("Done")

    return rs, R


# Saving positions
rscalc, R = Verlet(r, v)

np.save("Earth-Moon/R", R)
np.save("Earth-Moon/rs", rscalc)

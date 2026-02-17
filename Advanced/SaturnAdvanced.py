import numpy as np
from numba import njit
from numba_progress import ProgressBar

# Simulation parameters
dt, steps = 100, int(5e3)
downsample = 1  # Downsample points calculated to points recorded
N = 1000  # No. light bodies

print("Elapsed simulated time: " + str(dt * steps / 31556952) + " years")

# Massive bodies boundary conditions
M = (
    np.array(
        [
            568.32 * 1e4,
            0.379,
            1.08,
            6.18,
            11,
            23.1,
            1345.5,
            0.056,
            18.1,
            0.00005,
            0.000001,
            0.00007,
            0.0016,
            0.0014,
            0.0053,
            0.019,
            0.00004,
            0.00007,
            0.0003,
            0.083,
        ]
    )
    * 1e20
)

R0 = (
    np.array(
        [
            0,
            185.52,
            238.02,
            294.66,
            377.4,
            527.04,
            1221.87,
            1500.93,
            3560.85,
            133.583,
            136.5,
            137.67,
            139.353,
            141.7,
            151.422,
            151.472,
            294.72,
            294.72,
            377.44,
            12944,
        ]
    )
) * 1e6

omega = np.zeros_like(R0)
omega[1:] = np.sqrt(6.67e-11 * M[0] / R0[1:] ** 3)


# Light bodies boundary conditions
theta = 2 * np.pi * np.random.random(N)


# Astroid positions (even density in caertesian space)
r_max = 140.220e6  # Extent of Saturn's rings
r_min = 74.500e6  # Diameter of Saturn


noise = r_min + (r_max - r_min) * np.random.random(N)[:, None]


omega_r = np.sqrt(6.67e-11 * M[0] / noise**3)

r = np.ones((N, 2)) * noise
r[:, 0] *= np.cos(theta)
r[:, 1] *= np.sin(theta)

v = np.ones((N, 2)) * omega_r * noise
v[:, 0] *= -np.sin(theta)
v[:, 1] *= np.cos(theta)


# Acceleration of light bodies
@njit(nogil=True, cache=True)
def acc(r, R):
    diff = R[None, :, :] - r[:, None, :]
    dx, dy = diff[..., 0], diff[..., 1]
    dist = np.sqrt(dx * dx + dy * dy + 1e8)  # Softened gravity
    isl = diff / (dist[..., None] ** 3)
    a = 6.67e-11 * np.sum(M[None, :, None] * isl, axis=1)
    return a


# r += dr ()
@njit(nogil=True, cache=True)
def Kahan(r, dr, c):
    y = dr - c
    t = r + y
    z = t.copy() - r
    c = z.copy() - y
    r = t.copy()
    return r, c


# Numerical integrator
@njit(nogil=True, cache=True)
def ForestRuth(r, v, progress_proxy):
    # Setting up Kahan
    cr = np.zeros_like(r)
    cv = np.zeros_like(v)

    # Setting up FR

    c1 = 1 / (2 * (2 - 2 ** (1 / 3)))
    c2 = (1 - 2 ** (1 / 3)) / (2 * (2 - 2 ** (1 / 3)))
    c3 = c2
    c4 = c1

    d1 = 1 / (2 - 2 ** (1 / 3))
    d2 = -(2 ** (1 / 3)) / (2 - 2 ** (1 / 3))
    d3 = d1

    # History
    rs = np.empty((int(steps / downsample), N, 2))
    rs[0] = r.copy()

    R = np.empty((int(steps / downsample), len(M), 2))
    R[0, :, 0] = R0

    # Substep initialisation
    R_sub = np.empty((downsample, len(M), 2))
    dt_sub = np.arange(downsample) * dt

    progress_proxy.update(1)

    for i in range(1, int(steps / downsample)):
        # Heavy positions
        t_sub = dt_sub + (i - 1) * dt * downsample
        R_sub[..., 0] = R0[None, :] * np.cos(omega * t_sub[:, None])
        R_sub[..., 1] = R0[None, :] * np.sin(omega * t_sub[:, None])

        # Light positions
        for j in range(downsample):
            a = acc(r, R_sub[j])
            v, cv = Kahan(v, (dt * c1) * a, cv)
            r, cr = Kahan(r, (dt * d1) * v, cr)

            a = acc(r, R_sub[j])
            v, cv = Kahan(v, (dt * c2) * a, cv)
            r, cr = Kahan(r, (dt * d2) * v, cr)

            a = acc(r, R_sub[j])
            v, cv = Kahan(v, (dt * c3) * a, cv)
            r, cr = Kahan(r, (dt * d3) * v, cr)

            a = acc(r, R_sub[j])
            v, cv = Kahan(v, (dt * c4) * a, cv)

        # Storing positions
        rs[i] = r.copy()
        R[i] = R_sub[-1].copy()

        progress_proxy.update(1)

    print("Done")

    return rs, R


with ProgressBar(total=int(steps / downsample)) as progress:
    rs, R = ForestRuth(r, v, progress)


# Saving positions to directory
np.save("Advanced/RSaturn", R)
np.save("Advanced/rsSaturn", rs)

print("Saved")

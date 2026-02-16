import numpy as np
from tqdm import tqdm

# Simulation parameters
dt, steps = 100, int(5e7)
downsample = int(1e2)  # Downsample points calculated to points recorded
N = 1000    # No. light bodies

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
def acc(r, R):
    diff = R[None, :, :] - r[:, None, :]
    dist = np.linalg.norm(diff, axis=2)
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

    R = np.empty((int(steps / downsample), len(M), 2))
    R[0, :, 0] = R0

    # Substep initialisation
    R_sub = np.empty((downsample, len(M), 2))
    dt_sub = np.linspace(0, dt * downsample, downsample)

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


rs, R = Verlet(r, v)


# Saving positions to directory
np.save("Saturn/RSaturn", R)
np.save("Saturn/rsSaturn", rs)

print("Saved")

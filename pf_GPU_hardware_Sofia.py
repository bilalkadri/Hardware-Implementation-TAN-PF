
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import norm
import time
from numba import njit,prange

# Load datasets
measurements = loadmat('Measurements.mat')
dem_heights = loadmat('DEM_heights.mat')
dem_complete = loadmat('DEM_Complete.mat')
data_v4 = loadmat('DataV4.mat')
proc_noise = loadmat('ProcNoise2.mat')

# Extract required data
h_baro = measurements['h_baro'].flatten()
h_radar = measurements['h_radar'].flatten()
h_db = h_baro[699:750] - h_radar[699:750]

# # Plot h_db
# plt.figure()
# plt.plot(h_db, 'r')
# plt.grid(True)
# plt.title('Difference in Barometric and Radar Heights')
# plt.show()

# # Plot histogram
# plt.figure()
# plt.hist(h_db, bins=10)
# plt.title('Histogram of h_db')
# plt.show()

# Initialize variables
N = 2000
particles = np.random.randn(2, N)
Pk = np.array([[1e-9, 1e-8], [1e-6, 2e-9]])
Qk = np.array([[0.1, 0], [0, 0.1]])
Rk = np.array([[0.04, 0], [0, 0.01]])
c = [Pk.copy() for _ in range(N)]

Lat_tercom = np.zeros(50)
Long_tercom = np.zeros(50)
Neff = np.zeros(50)
Xcorr = np.zeros((2, 50))

# Particle filter loop
# @njit(parallel=True)
for k in range(50):
    m = np.abs(dem_complete['Z'] - np.mean(np.abs(dem_complete['Z'] - h_db[k])))
    row, col = np.unravel_index(np.argmin(m), m.shape)
    pos = dem_complete['M'][row, col].flatten()
    Lat_tercom[k] = pos[0]
    Long_tercom[k] = pos[1]
    z = np.array([Lat_tercom[k], Long_tercom[k]])

    for i in range(N):
        start_execution_time=time.time()
        # Prediction Step
        Xpred = particles[:, i]
        Wk = proc_noise['P2'][k].flatten()[0] * np.random.randn(2)
        F = np.eye(2)
        Xpred = F @ Xpred + Wk

        Pk = c[i]
        Pk = F @ Pk @ F.T + Qk

        # Measurement Update
        H = np.eye(2)
        I = z - H @ Xpred
        S = H @ Pk @ H.T + Rk
        K = Pk @ H.T @ np.linalg.inv(S)

        Xupdt = Xpred + K @ I
        Pk = (np.eye(2) - K @ H) @ Pk

        c[i] = Pk
        particles[:, i] = Xupdt

    # Resampling
    if k == 0:
        dist = np.sqrt((particles[0, :] - z[0])**2 + (particles[1, :] - z[1])**2)
        w = 1 / dist
        w /= np.sum(w)

    Neff[k] = 1 / np.sum(w**2)
    if Neff[k] < 300:
        cdf = np.cumsum(w)
        new_particles = np.zeros_like(particles)
        for j in range(N):
            uj = np.random.uniform(0, 1 / N) + j / N
            idx = np.searchsorted(cdf, uj)
            new_particles[:, j] = particles[:, idx]
        particles = new_particles
        w = np.ones(N) / N

    Xcorr[:, k] = np.sum(particles * w, axis=1)


print("--- %s seconds ---" % (time.time() - start_execution_time))
# Calculate RMSE
xgps = data_v4['DataV4'][699:750, 7].flatten()
ygps = data_v4['DataV4'][699:750, 8].flatten()

rmse_x = np.sqrt(np.mean((ygps[:50] - Lat_tercom)**2))
rmse_y = np.sqrt(np.mean((xgps[:50] - Long_tercom)**2))

rmse_xpf = np.sqrt(np.mean((ygps[:50] - Xcorr[0, :50])**2))
rmse_ypf = np.sqrt(np.mean((xgps[:50] - Xcorr[1, :50])**2))

print(f"RMSE (x): {rmse_x}")
print(f"RMSE (y): {rmse_y}")
print(f"RMSE PF (x): {rmse_xpf}")
print(f"RMSE PF (y): {rmse_ypf}")
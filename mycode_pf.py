import numpy as np
import time
from numba import njit

# Load datasets (Assuming these have already been loaded)
measurements = loadmat('Measurements.mat')
dem_heights = loadmat('DEM_heights.mat')
dem_complete = loadmat('DEM_Complete.mat')
data_v4 = loadmat('DataV4.mat')
proc_noise = loadmat('ProcNoise2.mat')

# Extract required data
h_baro = measurements['h_baro'].flatten()
h_radar = measurements['h_radar'].flatten()
h_db = h_baro[699:750] - h_radar[699:750]

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

# Precompute constants
F = np.eye(2)
H = np.eye(2)
I2 = np.eye(2)

@njit(parallel=True)
def particle_filter(dem_complete, proc_noise, N, particles, c, h_db, F, H, I2, Rk, Qk, Lat_tercom, Long_tercom, Neff, Xcorr):
    for k in range(50):
        m = np.abs(dem_complete['Z'] - np.mean(np.abs(dem_complete['Z'] - h_db[k])))
        row, col = np.unravel_index(np.argmin(m), m.shape)
        pos = dem_complete['M'][row, col].flatten()
        Lat_tercom[k] = pos[0]
        Long_tercom[k] = pos[1]
        z = np.array([Lat_tercom[k], Long_tercom[k]])

        w = np.zeros(N)
        for i in range(N):
            # Prediction Step
            Xpred = particles[:, i]
            Wk = proc_noise['P2'][k].flatten()[0] * np.rand

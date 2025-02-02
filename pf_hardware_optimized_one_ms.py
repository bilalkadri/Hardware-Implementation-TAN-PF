import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import norm
import time
#import cuda


# Load datasets
measurements = loadmat('D:\PeerJ GPU Paper Python Code\Hardware-Implementation-TAN-PF\Measurements.mat')
dem_heights = loadmat('D:\PeerJ GPU Paper Python Code\Hardware-Implementation-TAN-PF\DEM_heights.mat')
dem_complete = loadmat('D:\PeerJ GPU Paper Python Code\Hardware-Implementation-TAN-PF\DEM_Complete.mat')
data_v4 = loadmat('D:\PeerJ GPU Paper Python Code\Hardware-Implementation-TAN-PF\DataV4.mat')
proc_noise = loadmat('Hardware-Implementation-TAN-PF/ProcNoise2.mat')

# Extract required data
h_baro = measurements['h_baro'].flatten()
h_radar = measurements['h_radar'].flatten()
h_db = h_baro[699:750] - h_radar[699:750]

print(type(measurements))
print(type(dem_heights))
print(type(dem_complete))
print(type(data_v4))
print(type(proc_noise))
print(type(h_baro))
print(type(h_radar))
print(type(h_db))

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

# Convert to ndarrays
dem_hgts= dem_heights['Z']
print(type(dem_hgts))
print(dem_hgts.shape)
dem_h = dem_hgts[0:3600,0:3600]
print(dem_h.shape)

#print(dem_complete)
dem_comp = dem_complete['M']
print(type(dem_comp))
print(dem_comp.shape)

print(dem_comp)

#print(data_v4)
Data = data_v4['DataV4']
print(type(Data))
print(Data.shape)

#print(proc_noise)
pr_noise = proc_noise['P2']
print(type(pr_noise))
print(pr_noise.shape)

#print(h_db.shape)
h_db_2=h_db[0:50]
print(h_db_2.shape)

from numba import njit, prange
import numpy as np

#@njit(nopython =True)
def particle_filter(dem_complete_Z, dem_complete_M, h_db, proc_noise_P2, particles, c, Qk, Rk, data_v4, N):
    # Initialize output arrays
    Lat_tercom = np.zeros(50)
    Long_tercom = np.zeros(50)
    Neff = np.zeros(50)
    Xcorr = np.zeros((2, 50))

    num_rows, num_cols = dem_complete_Z.shape  # Shape of the 2D array

    for k in prange(50):
        # Validate dimensions and input assumptions
        #if dem_complete_Z.shape != dem_complete_M[:, :, 0].shape:
            #raise ValueError("dem_complete_Z and dem_complete_M dimensions do not match.")
        #if len(h_db) < 50 or len(proc_noise_P2) < 50:
            #raise ValueError("h_db or proc_noise_P2 has insufficient length.")
        #if particles.shape != (2, N):
            #raise ValueError("particles array must have shape (2, N).")

        # Calculate matching metric
        m = np.abs(dem_complete_Z - np.mean(np.abs(dem_complete_Z - h_db[k])))
        idx = np.argmin(m)  # Find the flattened index of the minimum value
        row = idx // num_cols  # Convert to row index
        col = idx % num_cols  # Convert to column index

        pos = dem_complete_M[row, col].flatten()
        Lat_tercom[k] = pos[0]
        Long_tercom[k] = pos[1]
        z = np.array([Lat_tercom[k], Long_tercom[k]])

        # Particle filter loop
        for i in range(N):
            # Prediction step
            st = time.time()
            Xpred = particles[:, i]
            Wk = proc_noise_P2[k].flatten()[0] * np.random.randn(2)
            F = np.eye(2)
            Xpred = F @ Xpred + Wk

            Pk = c[i]
            Pk = F @ Pk @ F.T + Qk

            # Measurement update
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
            dist[dist == 0] = 1e-6  # Avoid division by zero
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

        # Compute corrected position
        Xcorr[:, k] = np.sum(particles * w, axis=1)
        et = time.time()
        print(et-st)
    # Calculate RMSE
    xgps = data_v4[699:750, 7].flatten()
    ygps = data_v4[699:750, 8].flatten()

    rmse_x = np.sqrt(np.mean((ygps[:50] - Lat_tercom)**2))
    rmse_y = np.sqrt(np.mean((xgps[:50] - Long_tercom)**2))
    rmse_xpf = np.sqrt(np.mean((ygps[:50] - Xcorr[0, :50])**2))
    rmse_ypf = np.sqrt(np.mean((xgps[:50] - Xcorr[1, :50])**2))

    return rmse_x, rmse_y, rmse_xpf, rmse_ypf

import numpy as np
import time

# Example input data
np.random.seed(42)  # For reproducibility

# Digital Elevation Model (DEM) data
dem_complete_Z = dem_h  # Random 2D elevation data
dem_complete_M = dem_comp  # Lat/Lon coordinate map corresponding to DEM

# Reference database and process noise
h_db = h_db  # Reference heights for matching
proc_noise_P2 = pr_noise  # Process noise values for each step

# Particles and covariance matrices
N = 1000  # Number of particles
particles = np.random.rand(2, N)  # Initial particle states (2D positions)
c = np.array([np.eye(2) for _ in range(N)])  # Covariance matrices for each particle

# Noise covariances
Qk = np.eye(2) * 0.01  # Process noise covariance
Rk = np.eye(2) * 0.05  # Measurement noise covariance

# Simulated GPS data
data_v4 = Data[:,1:10]  # Random data with at least 9 columns (x/y GPS in cols 7/8)

# Run the particle filter
#from particle_filter_module import particle_filter  # Assuming you've saved the function in a module
st = time.time()
rmse_x, rmse_y, rmse_xpf, rmse_ypf = particle_filter(
dem_complete_Z, dem_complete_M, h_db_2, proc_noise_P2,
particles, c, Qk, Rk, data_v4, N
)
et = time.time()
# Display results
print(f"RMSE (x): {rmse_x}")
print(f"RMSE (y): {rmse_y}")
print(f"RMSE PF (x): {rmse_xpf}")
print(f"RMSE PF (y): {rmse_ypf}")
print(et-st)
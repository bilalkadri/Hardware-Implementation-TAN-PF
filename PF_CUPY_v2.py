import numpy as np
#import matplotlib.pyplot as plt
#from scipy.io import loadmat
import scipy as sc
from scipy import io
from scipy.stats import norm
import time
#import cuda

import cupy as cp

#import cupy as cp
#x = cp.array([1, 2, 3])
#print(x)

# Load datasets
#measurements = loadmat('/home/oem/Hardware-Implementation-TAN-PF/Measurements.mat')
measurements = sc.io.loadmat('Measurements.mat')
#dem_heights = loadmat('/home/oem/Hardware-Implementation-TAN-PF/DEM_heights.mat')
dem_heights = sc.io.loadmat('DEM_heights.mat')
#dem_complete = loadmat('/home/oem/Hardware-Implementation-TAN-PF/DEM_Complete.mat')
dem_complete = sc.io.loadmat('DEM_Complete.mat')
#data_v4 = loadmat('/home/oem/Hardware-Implementation-TAN-PF/DataV4.mat')
data_v4 = sc.io.loadmat('DataV4.mat')
#proc_noise = loadmat('/home/oem/Hardware-Implementation-TAN-PF/ProcNoise2.mat')
proc_noise = sc.io.loadmat('ProcNoise2.mat')

# Extract required data
h_baro = measurements['h_baro'].flatten()
#h_baro = cp.asarray(h_baro)
h_radar = measurements['h_radar'].flatten()
#h_radar = cp.asarray(h_radar)
h_db = h_baro[699:750] - h_radar[699:750]
#h_db = cp.asarray(h_db)

print(type(measurements))
print(type(dem_heights))
print(type(dem_complete))
print(type(data_v4))
print(type(proc_noise))
print(type(h_baro))
print(type(h_radar))
print(type(h_db))

# Initialize variables
N = 6000
particles = np.random.randn(2, N)
particles = cp.asarray(particles)
Pk = np.array([[1e-9, 1e-8], [1e-6, 2e-9]])
Pk = cp.asarray(Pk)
Qk = np.array([[0.01, 0], [0, 0.01]])
Qk = cp.asarray(Qk)
Rk = np.array([[4, 0], [0, 1]])
Rk = cp.asarray(Rk)
c = [Pk.copy() for _ in range(N)]

Lat_tercom = np.zeros(50)
Lat_tercom = cp.asarray(Lat_tercom)
Long_tercom = np.zeros(50)
Long_tercom = cp.asarray(Long_tercom)
Neff = np.zeros(50)
Neff = cp.asarray(Neff)
Xcorr = np.zeros((2, 50))
Xcorr = cp.asarray(Xcorr)

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
Data = cp.asarray(Data)
print(type(Data))
print(Data.shape)

#print(proc_noise)
pr_noise = proc_noise['P2']
pr_noise = cp.asarray(pr_noise)
print(type(pr_noise))
print(pr_noise.shape)

#print(h_db.shape)
h_db_2=h_db[0:50]
print(h_db_2.shape)

from numba import njit, prange
import numpy as np
results = []
filename = "calculation_result_pf.txt"
#with open(filename, "w") as file:


#@njit(nopython =True)
def particle_filter(dem_complete_Z, dem_complete_M, h_db, proc_noise_P2, particles, c, Qk, Rk, data_v4, N):
    # Initialize output arrays
    Lat_tercom = np.zeros(50)
    Lat_tercom = cp.asarray(Lat_tercom)
    Long_tercom = np.zeros(50)
    Long_tercom = cp.asarray(Long_tercom)
    Neff = np.zeros(50)
    Neff = cp.asarray(Neff)
    Xcorr = np.zeros((2, 50))
    Xcorr = cp.asarray(Xcorr)
    #dem_complete_Z = cp.asarray(dem_complete_Z)
    num_rows, num_cols = dem_complete_Z.shape  # Shape of the 2D array

#with open(filename, "w") as file:

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
        Lat_tercom[k] = cp.asarray(pos[0])
        Long_tercom[k] = cp.asarray(pos[1])
        z = cp.array([Lat_tercom[k], Long_tercom[k]])

        # Particle filter loop
        for i in range(N):
            # Prediction step
            st = time.time()
            Xpred = particles[:, i]
            Wk = proc_noise_P2[k].flatten()[0] * cp.random.randn(2)
            F = cp.eye(2)
            Xpred = F @ Xpred + Wk

            Pk = c[i]
            Pk = F @ Pk @ F.T + Qk

            # Measurement update
            H = cp.eye(2)
            I = z - H @ Xpred
            S = H @ Pk @ H.T + Rk
            K = Pk @ H.T @ cp.linalg.inv(S)

            Xupdt = Xpred + K @ I
            Pk = (cp.eye(2) - K @ H) @ Pk

            c[i] = Pk
            particles[:, i] = Xupdt

        # Resampling
        if k == 0:
            dist = cp.sqrt((particles[0, :] - z[0])**2 + (particles[1, :] - z[1])**2)
            dist[dist == 0] = 1e-6  # Avoid division by zero
        w = 1 / dist
        w /= cp.sum(w)

        Neff[k] = 1 / cp.sum(w**2)
        if Neff[k] < 300:
            cdf = cp.cumsum(w)
            new_particles = np.zeros_like(particles)
            for j in range(N):
                uj = cp.random.uniform(0, 1 / N) + j / N
                idx = cp.searchsorted(cdf, uj)
                new_particles[:, j] = particles[:, idx]
            particles = new_particles
            w = cp.ones(N) / N

        # Compute corrected position
        Xcorr[:, k] = cp.sum(particles * w, axis=1)
        et = time.time()
        t = et-st
        results.append(t)
        #file.write(f"The execution time per data point is {t}\n")
        #print(et-st)
    # Calculate RMSE
    xgps = data_v4[699:750, 7].flatten()
    ygps = data_v4[699:750, 8].flatten()

    rmse_x = cp.sqrt(cp.mean((xgps[:50] - Lat_tercom)**2))
    rmse_y = cp.sqrt(cp.mean((ygps[:50] - Long_tercom)**2))
    rmse_xpf =cp.sqrt(cp.mean((xgps[:50] - Xcorr[0, :50])**2))
    rmse_ypf = cp.sqrt(cp.mean((ygps[:50] - Xcorr[1, :50])**2))
    Xcorr_Lat = Xcorr[0, :50]
    Xcorr_Lon = Xcorr[1,:50]
    return rmse_x, rmse_y, rmse_xpf, rmse_ypf,results,xgps,ygps,Xcorr_Lat,Xcorr_Lon

import numpy as np
import time

# Example input data
cp.random.seed(42)  # For reproducibility

# Digital Elevation Model (DEM) data
dem_complete_Z = dem_h  # Random 2D elevation data
#dem_complete_Z = cp.asarray(dem_complete_Z)
dem_complete_M = dem_comp  # Lat/Lon coordinate map corresponding to DEM

# Reference database and process noise
h_db = h_db  # Reference heights for matching
proc_noise_P2 = pr_noise  # Process noise values for each step

# Particles and covariance matrices
N = 1000  # Number of particles
result2 = []
result3 = []
result4 = []
N = [1000,2000,3000,4000,5000]
for i in range(len(N)):
 particles = cp.random.rand(2, N[i])  # Initial particle states (2D positions)
 c = cp.array([cp.eye(2) for _ in range(N[i])])  # Covariance matrices for each particle

# Noise covariances
 Qk = cp.eye(2) * 0.01  # Process noise covariance
 Rk = cp.eye(2) * 0.05  # Measurement noise covariance

# Simulated GPS data
 data_v4 = Data[:,1:10]  # Random data with at least 9 columns (x/y GPS in cols 7/8)
 data_v4 = cp.asarray(data_v4)

# Run the particle filter
#from particle_filter_module import particle_filter  # Assuming you've saved the function in a module
#N = [1000,2000]
 st = time.time()

#with open(results.txt, "w") as file:
 rmse_x, rmse_y, rmse_xpf, rmse_ypf,results,xgps,ygps,Xcorr_Lat,Xcorr_Lon= particle_filter(
dem_complete_Z, dem_complete_M, h_db_2, proc_noise_P2,
particles, c, Qk, Rk, data_v4, N[i]
)
 et = time.time()
 result2.append(et-st)
 result3.append(rmse_xpf)
 result4.append(rmse_ypf)
# Display results
print(f"RMSE (x): {rmse_x}")
print(f"RMSE (y): {rmse_y}")
print(f"RMSE PF (x): {rmse_xpf}")
print(f"RMSE PF (y): {rmse_ypf}")
print(et-st)
N = [1000,2000,3000,4000,5000,6000,7000,8000]
with open("resultspf.txt", "w") as file:
 file.write(f"The execution time per data point is: {results}\n") 
 file.write(f"The values of particle number are taken from {N}\n") 
 #file.write(f"The corresponding execution time is {et-st}\n") 
 file.write(f"The corresponding execution time is {result2}\n") 
 file.write(f"RMSE (x): {result3}\n")
 file.write(f"RMSE (y): {result4}\n")
 file.write(f"Latitude gps is: {xgps}\n")
 file.write(f"Longitude gps is: {ygps}\n")
 file.write(f"Predicted Latitude is: {Xcorr_Lat}\n")
 file.write(f"Predicted Longitude is: {Xcorr_Lon}\n")
 #file.write(f"The rmse Longitude is: {Xcorr_Lon}\n")
 #file.write(f"The rmse Latitude is: {Xcorr_Lon}\n")


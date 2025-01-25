import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time

# --- Load your .mat data files ---
# Adjust file paths as necessary.
measurements = loadmat('Measurements.mat')
dem_heights  = loadmat('DEM_heights.mat')
dem_complete = loadmat('DEM_Complete.mat')
data_v4      = loadmat('DataV4.mat')
proc_noise2  = loadmat('ProcNoise2.mat')

# Suppose these variables exist in the loaded .mat files:
# h_baro, h_radar, Z, M, DataV4, P2, etc.
# You MUST inspect each .mat file’s structure to extract 
# the right arrays. For example:
h_baro = measurements['h_baro'].flatten()   # example: shape (>=750,)
h_radar = measurements['h_radar'].flatten() # example
Z = dem_heights['Z']        # example: shape depends on your .mat
M = dem_complete['M']       # might be a cell-like structure in MATLAB
DataV4_arr = data_v4['DataV4']
P2 = proc_noise2['P2'].flatten()  # example

# --------------------------------------------------------
# Replicate the same indexing as in MATLAB:
# In MATLAB, h_db = h_baro(700:750) - h_radar(700:750) 
# That slice is 51 elements (700..750 inclusive).
# In Python, to get 51 elements from index 700..750:
h_db = h_baro[700:751] - h_radar[700:751]

# Quick plot of h_db (like MATLAB figure, hist)
plt.figure()
plt.plot(h_db, 'r')
plt.title('h\\_db')
plt.grid(True)
plt.show()

plt.figure()
plt.hist(h_db, bins=10)
plt.title('Histogram of h\\_db')
plt.show()

# --------------------------------------------------------
# Particle Filter Initialization
N = 2000  # Number of particles
particles = np.random.randn(2, N)  # row1 -> x-pos, row2 -> y-pos

# Covariance matrix Pk (initial)
Pk_init = np.array([[1e-9, 1e-8],
                    [1e-6, 2e-9]])

# Store each particle’s covariance in a list (similar to cell array)
c_list = [Pk_init.copy() for _ in range(N)]

Qk = np.array([[0.1, 0.0],
               [0.0, 0.1]])  # Process noise covariance

Rk = np.array([[0.04, 0.0],
               [0.0,  0.01]])  # Measurement noise covariance

# Arrays to store TERCOM-based lat/long and PF-corrected states
Lat_tercom = np.zeros(50)
Long_tercom = np.zeros(50)
Xcorr = np.zeros((2, 50))  # [ [x1,..,x50], [y1,..,y50] ]

# --------------------------------------------------------
# Timing start
t_start = time.time()

# Main loop for k = 1..50 in MATLAB => range(0,50) in Python
for k in range(50):

    # ----------------------------------------------------
    # "Measurement" from TERCOM-like approach (just mirroring your code):
    # m = abs(Z - mean(mean(abs(Z - h_db(k)))))
    # We'll replicate the logic carefully. 
    # NOTE: This depends on shapes of Z and how h_db(k) is used.
    # In MATLAB: m = abs(Z - mean(mean(abs(Z - h_db(k)))))
    # You might need to adapt if Z is 2D or 1D in Python.

    # Example approach (assuming Z is 2D):
    diff_2d = np.abs(Z - h_db[k])    # shape same as Z
    mean_val = np.mean(diff_2d)      # mean of entire 2D array
    m = np.abs(Z - mean_val)         # replicate m

    # Find the minimum entry in m:
    row, col = np.unravel_index(np.argmin(m), m.shape)

    # Retrieve position from M (which might be a cell array in MATLAB).
    # In Python, if M is an object array: M[row,col] could be e.g. [lat, long]
    pos = M[row, col]  # You may need to adjust syntax if M is a nested structure
    # Assume pos is [lat, long]
    Lat_tercom[k] = pos[0]
    Long_tercom[k] = pos[1]

    # The measurement vector z:
    z = np.array([Lat_tercom[k], Long_tercom[k]])  # shape (2,)

    # ----------------------------------------------------
    # For each particle, do prediction + update
    Xpred = np.zeros((2, N))
    Xupdt = np.zeros((2, N))
    I_mat = np.zeros((2, N))

    for i in range(N):

        # ---------- Prediction ----------
        Xpred[:, i] = particles[:, i].copy()

        # Process noise: Wk
        # Wk = P2[k]*randn(2,1) in MATLAB => shape (2,)
        Wk = P2[k] * np.random.randn(2)

        # Define the (linear) process model: F = I2
        F = np.eye(2)
        # Predict state
        Xpred[:, i] = F @ Xpred[:, i] + Wk

        # Predict covariance
        Pk_i = c_list[i]
        Pk_i = F @ Pk_i @ F.T + Qk

        # ---------- Measurement Update ----------
        # h matrix => Identity(2)
        H = np.eye(2)

        # Innovation I = z - H*Xpred
        I_mat[:, i] = z - (H @ Xpred[:, i])

        # S = H*Pk_i*H' + Rk
        S = H @ Pk_i @ H.T + Rk

        # K = Pk_i*H'*inv(S)
        K = Pk_i @ H.T @ np.linalg.inv(S)

        # Xupdt = Xpred + K*Innovation
        Xupdt[:, i] = Xpred[:, i] + K @ I_mat[:, i]

        # Update covariance: Pk = (I-KH)*Pk
        Pk_i = (np.eye(2) - K @ H) @ Pk_i

        # Store updated covariance
        c_list[i] = Pk_i

        # Update the particle with the corrected state
        particles[:, i] = Xupdt[:, i]

    # ----------------------------------------------------
    # Weighting (only at first iteration in the MATLAB code)
    if k == 0:
        dist = np.sqrt((Xupdt[0, :] - z[0])**2 + (Xupdt[1, :] - z[1])**2)
        w = 1.0 / dist
        # Normalize
        w = w / np.sum(w)

    # ----------------------------------------------------
    # Check Neff for resampling
    Neff = 1.0 / np.sum(w**2)
    Thresh = 300

    if Neff < Thresh:
        # --- Resampling ---
        cdf = np.cumsum(w)
        # Using systematic resampling approach:
        # u1 = random in [0, 1/N]
        u1 = (1.0 / N) * np.random.rand()

        i = 0
        newX = np.zeros((2, N))
        for j in range(N):
            uj = u1 + (j / N)
            while (uj > cdf[i]) and (i < N - 1):
                i += 1
            # Copy the i-th particle state
            newX[:, j] = Xupdt[:, i]

        # Reinitialize uniform weights
        w = np.ones(N) * (1.0 / N)
        # Replace Xupdt with newX
        Xupdt = newX

    # ----------------------------------------------------
    # Compute an estimate from the updated set of particles
    # The code uses sum(Xupdt.*w,2) => weighted sum
    # shape Xupdt => (2, N), w => (N,)
    Xcorr[:, k] = np.sum(Xupdt * w, axis=1)

    # End of time-step k
    print(f"Iteration k = {k+1}")

# End main loop

# Timing
elapsed_time = time.time() - t_start
print("Elapsed time:", elapsed_time, "seconds")

# --------------------------------------------------------
# Compare with “ground truth”
# xgps, ygps from your DataV4 (MATLAB: DataV4(700:750,8) => Python: [700:751, 7], 50 elements)
xgps = DataV4_arr[700:750, 7]  # careful with shapes
ygps = DataV4_arr[700:750, 8]

# MATLAB code:
# rmse_x = sqrt(mean((ygps(1:50)-Lat_tercom(1:50)').^2))
rmse_x = np.sqrt(np.mean((ygps[:50] - Lat_tercom[:50])**2))
rmse_y = np.sqrt(np.mean((xgps[:50] - Long_tercom[:50])**2))

rmse_xpf = np.sqrt(np.mean((ygps[:50] - Xcorr[0, :50])**2))
rmse_ypf = np.sqrt(np.mean((xgps[:50] - Xcorr[1, :50])**2))

print("RMSE_x (TERCOM) =", rmse_x)
print("RMSE_y (TERCOM) =", rmse_y)
print("RMSE_xpf (PF)   =", rmse_xpf)
print("RMSE_ypf (PF)   =", rmse_ypf)

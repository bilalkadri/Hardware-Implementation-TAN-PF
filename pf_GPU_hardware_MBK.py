import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from numba import njit, prange

# ---------------------------------------------------------------------
# 1) LOAD AND PREP DATA (plain NumPy arrays, no dictionaries in JIT)
# ---------------------------------------------------------------------
def load_and_prepare_data():
    # Load .mat files
    measurements = loadmat('Measurements.mat')
    dem_heights  = loadmat('DEM_heights.mat')
    dem_complete = loadmat('DEM_Complete.mat')
    data_v4      = loadmat('DataV4.mat')
    proc_noise   = loadmat('ProcNoise2.mat')

    # Extract needed arrays
    h_baro  = measurements['h_baro'].flatten()
    h_radar = measurements['h_radar'].flatten()
    # Slice from index 699..749 (51 elements in MATLAB)
    # In Python, 699:750 => 51 elements
    h_db = h_baro[699:750] - h_radar[699:750]

    # dem_complete['Z'] should be a 2D array, dem_complete['M'] presumably 2D/3D
    # Inspect your .mat for exact shapes. Suppose:
    # Z = dem_complete['Z']   # shape e.g. (R, C)
    # M = dem_complete['M']   # shape e.g. (R, C), each cell = [lat, lon], or (R, C, 2)

    Z_mat = dem_complete['Z']  # might be object dtype
    Z = np.ascontiguousarray(Z_mat, dtype=np.float64)

    M_mat = dem_complete['M']  # Possibly shape (rows, cols), each cell an array of shape (1,2) or similar

    rows, cols = M_mat.shape
    M = np.empty((rows, cols, 2), dtype=np.float64)

    for r in range(rows):
        for c in range(cols):
            val = M_mat[r, c]               # some array, maybe shape (1,2) or (2,1)
            val = np.array(val).ravel()     # ensure shape (2,) => [lat, lon]
            M[r, c, 0] = val[0]             # lat
            M[r, c, 1] = val[1]             # lon

    M = np.ascontiguousarray(M, dtype=np.float64)


    # Similarly, proc_noise['P2'][k] is your process noise scale
    P2 = proc_noise['P2'].flatten()

    # GPS data from data_v4
    # data_v4['DataV4'] => shape (>=750, >=9)
    # Indices: [699:750, 7] => xgps, [699:750, 8] => ygps
    DataV4_arr = data_v4['DataV4']
    xgps = DataV4_arr[699:750, 7].flatten()
    ygps = DataV4_arr[699:750, 8].flatten()

    return h_db, Z, M, P2, xgps, ygps


# ---------------------------------------------------------------------
# 2) HELPER FUNCTIONS FOR Numba
# ---------------------------------------------------------------------
@njit(parallel=False)
def manual_argmin_2d(array2d):
    """
    Manual argmin of a 2D NumPy array for Numba compatibility.
    Returns (row, col) for the minimum value.
    """
    rows, cols = array2d.shape
    min_val = array2d[0, 0]
    min_idx = 0
    for r in range(rows):
        for c in range(cols):
            val = array2d[r, c]
            if val < min_val:
                min_val = val
                min_idx = r * cols + c
    row = min_idx // cols
    col = min_idx % cols
    return row, col

# ---------------------------------------------------------------------
# 3) MAIN PARTICLE FILTER FUNCTION (JIT-compiled)
# ---------------------------------------------------------------------
@njit(parallel=True)  # We'll parallelize the inner loop with prange
def particle_filter(
    N, steps,
    h_db,               # 1D array of shape (steps,)
    Z,                  # 2D array, shape (R, C)
    M,                  # Suppose shape (R, C, 2)
    P2,                 # 1D array of shape (steps,) for noise scaling
    particles,          # 2xN array
    covariances,        # Nx2x2 array for each particle's covariance
    Qk,                 # 2x2
    Rk,                 # 2x2
):
    """
    Runs the particle filter for 'steps' iterations, each with 'N' particles.
    Returns: Lat_tercom, Long_tercom, Xcorr, Neff
    """
    Lat_tercom = np.zeros(steps)
    Long_tercom = np.zeros(steps)
    Xcorr = np.zeros((2, steps))
    Neff_arr = np.zeros(steps)

    # Weights (initialized uniformly, updated after first measurement)
    w = np.ones(N, dtype=np.float64)
    w /= N

    rows, cols = Z.shape

    for k in range(steps):
        # 1) Terrain-based measurement
        # m = abs(Z - mean(abs(Z - h_db[k])))
        # Step A: compute temp_arr = abs(Z - h_db[k])
        temp_arr = np.empty_like(Z)
        val_k = h_db[k]
        for rr in range(rows):
            for cc in range(cols):
                diff = Z[rr, cc] - val_k
                if diff < 0:
                    diff = -diff
                temp_arr[rr, cc] = diff

        # Step B: take mean of temp_arr
        sum_val = 0.0
        for rr in range(rows):
            for cc in range(cols):
                sum_val += temp_arr[rr, cc]
        mean_diff = sum_val / (rows * cols)

        # Step C: m = abs(Z - mean_diff)
        m_arr = np.empty_like(Z)
        for rr in range(rows):
            for cc in range(cols):
                val = Z[rr, cc] - mean_diff
                if val < 0:
                    val = -val
                m_arr[rr, cc] = val

        # Step D: row,col = argmin(m_arr)
        row, col = manual_argmin_2d(m_arr)

        # pos = M[row,col] => e.g. [lat, lon]
        pos_lat = M[row, col, 0]
        pos_lon = M[row, col, 1]
        Lat_tercom[k] = pos_lat
        Long_tercom[k] = pos_lon

        z0 = pos_lat
        z1 = pos_lon

        # 2) Particle Filter update
        #    We parallelize over i in [0..N)
        for i in prange(N):
            # PREDICTION
            Xpred0 = particles[0, i]
            Xpred1 = particles[1, i]

            # Wk = P2[k]*randn(2)
            wk0 = P2[k] * np.random.randn()
            wk1 = P2[k] * np.random.randn()

            Xpred0 += wk0
            Xpred1 += wk1

            # Cov update: Pk += Qk (since F=I)
            Pk_i = covariances[i]
            Pk_i[0, 0] += Qk[0, 0]
            Pk_i[0, 1] += Qk[0, 1]
            Pk_i[1, 0] += Qk[1, 0]
            Pk_i[1, 1] += Qk[1, 1]

            # MEASUREMENT UPDATE (H=I => I=z - Xpred)
            I0 = z0 - Xpred0
            I1 = z1 - Xpred1

            # S = Pk_i + Rk
            S00 = Pk_i[0, 0] + Rk[0, 0]
            S01 = Pk_i[0, 1] + Rk[0, 1]
            S10 = Pk_i[1, 0] + Rk[1, 0]
            S11 = Pk_i[1, 1] + Rk[1, 1]

            # Invert S (2x2)
            detS = S00 * S11 - S01 * S10
            invS00 = S11 / detS
            invS01 = -S01 / detS
            invS10 = -S10 / detS
            invS11 = S00 / detS

            # K = Pk_i * invS
            K00 = Pk_i[0, 0] * invS00 + Pk_i[0, 1] * invS10
            K01 = Pk_i[0, 0] * invS01 + Pk_i[0, 1] * invS11
            K10 = Pk_i[1, 0] * invS00 + Pk_i[1, 1] * invS10
            K11 = Pk_i[1, 0] * invS01 + Pk_i[1, 1] * invS11

            # Xupdt = Xpred + K*I
            Xupdt0 = Xpred0 + (K00 * I0 + K01 * I1)
            Xupdt1 = Xpred1 + (K10 * I0 + K11 * I1)

            # Pk = (I - K) Pk if H=I
            t00 = 1.0 - K00
            t01 = -K01
            t10 = -K10
            t11 = 1.0 - K11

            newP00 = t00 * Pk_i[0, 0] + t01 * Pk_i[1, 0]
            newP01 = t00 * Pk_i[0, 1] + t01 * Pk_i[1, 1]
            newP10 = t10 * Pk_i[0, 0] + t11 * Pk_i[1, 0]
            newP11 = t10 * Pk_i[0, 1] + t11 * Pk_i[1, 1]

            Pk_i[0, 0] = newP00
            Pk_i[0, 1] = newP01
            Pk_i[1, 0] = newP10
            Pk_i[1, 1] = newP11

            # Store updated
            particles[0, i] = Xupdt0
            particles[1, i] = Xupdt1
            covariances[i] = Pk_i

        # 3) Initialize or update weights
        if k == 0:
            # w = 1 / dist => dist= sqrt((x - z0)^2 + (y - z1)^2)
            sum_w = 0.0
            for i in range(N):
                dx = particles[0, i] - z0
                dy = particles[1, i] - z1
                dist_i = np.sqrt(dx*dx + dy*dy)
                # Avoid division by zero
                w[i] = 1.0 / (dist_i + 1e-12)
                sum_w += w[i]
            # Normalize
            for i in range(N):
                w[i] /= sum_w

        # 4) Neff
        sum_w2 = 0.0
        for i in range(N):
            sum_w2 += w[i]*w[i]
        Neff_val = 1.0 / sum_w2
        Neff_arr[k] = Neff_val

        # 5) Resampling if needed
        if Neff_val < 300.0:
            # Build CDF
            cdf_arr = np.empty(N)
            cdf_arr[0] = w[0]
            for i in range(1, N):
                cdf_arr[i] = cdf_arr[i - 1] + w[i]

            new_particles = np.zeros_like(particles)
            for j in range(N):
                uj = np.random.uniform(0.0, 1.0/N) + j/N
                idx = np.searchsorted(cdf_arr, uj)
                if idx >= N:
                    idx = N - 1
                new_particles[0, j] = particles[0, idx]
                new_particles[1, j] = particles[1, idx]
            particles = new_particles

            # Reset weights
            uniform_val = 1.0 / N
            for i in range(N):
                w[i] = uniform_val

        # 6) Xcorr[:, k] = sum( particles * w )
        x_sum = 0.0
        y_sum = 0.0
        for i in range(N):
            x_sum += particles[0, i] * w[i]
            y_sum += particles[1, i] * w[i]
        Xcorr[0, k] = x_sum
        Xcorr[1, k] = y_sum

    return Lat_tercom, Long_tercom, Xcorr, Neff_arr


# ---------------------------------------------------------------------
# 4) MAIN / DRIVER SCRIPT
# ---------------------------------------------------------------------
def main():
    # 1) Load data from .mat files
    h_db, Z, M, P2, xgps, ygps = load_and_prepare_data()

    # 2) Display h_db and histogram (replicating your plots)
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(h_db, 'r')
    # plt.grid(True)
    # plt.title('Difference in Barometric and Radar Heights')
    # plt.show()

    # plt.figure()
    # plt.hist(h_db, bins=10)
    # plt.title('Histogram of h_db')
    # plt.show()

    # 3) Initialize Particle Filter variables
    N = 2000
    steps = 50

    particles = np.random.randn(2, N)
    # Covariance array shape (N,2,2)
    covariances = np.zeros((N, 2, 2))
    for i in range(N):
        covariances[i, 0, 0] = 1e-9
        covariances[i, 0, 1] = 1e-8
        covariances[i, 1, 0] = 1e-6
        covariances[i, 1, 1] = 2e-9

    Qk = np.array([[0.1, 0.0],
                   [0.0, 0.1]])
    Rk = np.array([[0.04, 0.0],
                   [0.0, 0.01]])

    # 4) Run Particle Filter
    t_start = time.time()
    Lat_tercom, Long_tercom, Xcorr, Neff = particle_filter(
        N, steps,
        h_db, Z, M, P2,
        particles, covariances,
        Qk, Rk
    )
    elapsed = time.time() - t_start
    print(f"Particle Filter completed in {elapsed:.4f} seconds")

    # 5) Compute RMSE
    rmse_x = np.sqrt(np.mean((ygps[:steps] - Lat_tercom)**2))
    rmse_y = np.sqrt(np.mean((xgps[:steps] - Long_tercom)**2))

    rmse_xpf = np.sqrt(np.mean((ygps[:steps] - Xcorr[0, :steps])**2))
    rmse_ypf = np.sqrt(np.mean((xgps[:steps] - Xcorr[1, :steps])**2))

    print(f"RMSE (x):    {rmse_x}")
    print(f"RMSE (y):    {rmse_y}")
    print(f"RMSE PF (x): {rmse_xpf}")
    print(f"RMSE PF (y): {rmse_ypf}")

if __name__ == "__main__":
    import time
    main()

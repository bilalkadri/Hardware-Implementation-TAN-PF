import numpy as np
from numba import njit, prange

@njit(parallel=True)
def particle_filter_numba(dem_complete_Z, dem_complete_M, h_db, proc_noise_P2,
                          particles, covariances, Qk, Rk, xgps, ygps, N):
    """
    A Numba-friendly particle filter function.

    Parameters
    ----------
    dem_complete_Z : 2D float64 array
        DEM elevations, shape (num_rows, num_cols).
    dem_complete_M : 3D float64 array
        DEM lat/lon array, shape (num_rows, num_cols, 2).
        dem_complete_M[i, j, 0] = latitude
        dem_complete_M[i, j, 1] = longitude
    h_db : 1D float64 array
        Reference heights for each iteration (length >= 50).
    proc_noise_P2 : 2D float64 array
        Process noise for each step, shape (>=50, something).
        This example uses proc_noise_P2[k, 0] as a scalar scale for noise.
    particles : 2D float64 array
        Initial particle states, shape (2, N).
    covariances : 3D float64 array
        Covariance matrices for each particle, shape (N, 2, 2).
    Qk : 2D float64 array
        Process noise covariance, shape (2, 2).
    Rk : 2D float64 array
        Measurement noise covariance, shape (2, 2).
    xgps : 1D float64 array
        Ground-truth x positions (GPS) for RMSE calculation (length >= 50).
    ygps : 1D float64 array
        Ground-truth y positions (GPS) for RMSE calculation (length >= 50).
    N : int
        Number of particles.

    Returns
    -------
    rmse_x : float64
        RMSE in x direction (DEM-based).
    rmse_y : float64
        RMSE in y direction (DEM-based).
    rmse_xpf : float64
        RMSE in x direction (particle-filter-based).
    rmse_ypf : float64
        RMSE in y direction (particle-filter-based).
    """

    # Output arrays
    Lat_tercom = np.zeros(50, dtype=np.float64)
    Long_tercom = np.zeros(50, dtype=np.float64)
    Neff_array = np.zeros(50, dtype=np.float64)
    Xcorr = np.zeros((2, 50), dtype=np.float64)

    # Preallocate weights
    w = np.ones(N, dtype=np.float64)  # Will be updated when k == 0

    num_rows = dem_complete_Z.shape[0]
    num_cols = dem_complete_Z.shape[1]

    # Identity and eye for updates
    I2 = np.eye(2, dtype=np.float64)

    for k in prange(50):

        # --- TERRAIN MATCHING STEP ---
        # Calculate "matching metric" for each cell. For demonstration:
        # m(i,j) = abs(DEM(i,j) - mean( abs(DEM - h_db[k]) ))
        # Then find (row,col) of minimum m.
        # NOTE: This is just an example; adapt to your actual matching logic.
        mean_diff = 0.0
        # Compute the average difference once
        for rr in range(num_rows):
            for cc in range(num_cols):
                mean_diff += abs(dem_complete_Z[rr, cc] - h_db[k])
        mean_diff /= (num_rows * num_cols)

        # Now compute metric "m"
        min_val = 1e15
        min_idx = 0
        for rr in range(num_rows):
            for cc in range(num_cols):
                val = abs(dem_complete_Z[rr, cc] - mean_diff)
                if val < min_val:
                    min_val = val
                    min_idx = rr * num_cols + cc

        row = min_idx // num_cols
        col = min_idx % num_cols

        # Extract lat/lon from dem_complete_M
        pos0 = dem_complete_M[row, col, 0]
        pos1 = dem_complete_M[row, col, 1]
        Lat_tercom[k] = pos0
        Long_tercom[k] = pos1

        # Observed measurement z = [lat, lon]
        z0 = pos0
        z1 = pos1

        # --- PARTICLE FILTER STEP ---
        # Prediction + Update for each particle
        for i in range(N):
            # Prediction: Xpred = F * Xprev + Wk
            # F is the identity in this example
            Xpred0 = particles[0, i]
            Xpred1 = particles[1, i]

            # Add process noise
            scale = proc_noise_P2[k, 0]  # e.g., shape (50, 1) or (50, something)
            # If np.random.randn() is not supported, see notes below
            noise0 = scale * np.random.randn()
            noise1 = scale * np.random.randn()

            Xpred0 += noise0
            Xpred1 += noise1

            # Covariance predict
            Pk = covariances[i, :, :]  # shape (2, 2)
            # Pk_new = I2 * Pk + Qk, but we want F@Pk@F.T + Qk; F=I => Pk+Qk
            # We'll do it manually to avoid Python object confusion:
            Pk[0, 0] = Pk[0, 0] + Qk[0, 0]
            Pk[0, 1] = Pk[0, 1] + Qk[0, 1]
            Pk[1, 0] = Pk[1, 0] + Qk[1, 0]
            Pk[1, 1] = Pk[1, 1] + Qk[1, 1]

            # Measurement update
            # z - H*Xpred => with H=I => (z0 - Xpred0, z1 - Xpred1)
            Innov0 = z0 - Xpred0
            Innov1 = z1 - Xpred1

            # S = Pk + Rk
            S00 = Pk[0, 0] + Rk[0, 0]
            S01 = Pk[0, 1] + Rk[0, 1]
            S10 = Pk[1, 0] + Rk[1, 0]
            S11 = Pk[1, 1] + Rk[1, 1]

            # invert S (2x2)
            detS = S00 * S11 - S01 * S10
            if abs(detS) < 1e-12:
                # to avoid singular
                S00 += 1e-6
                S11 += 1e-6
                detS = S00 * S11 - S01 * S10

            invS00 =  S11 / detS
            invS01 = -S01 / detS
            invS10 = -S10 / detS
            invS11 =  S00 / detS

            # Kalman gain K = Pk * inv(S)
            K00 = Pk[0, 0] * invS00 + Pk[0, 1] * invS10
            K01 = Pk[0, 0] * invS01 + Pk[0, 1] * invS11
            K10 = Pk[1, 0] * invS00 + Pk[1, 1] * invS10
            K11 = Pk[1, 0] * invS01 + Pk[1, 1] * invS11

            # Xupdt = Xpred + K*Innov
            Xupdt0 = Xpred0 + K00 * Innov0 + K01 * Innov1
            Xupdt1 = Xpred1 + K10 * Innov0 + K11 * Innov1

            # Pk = (I - K*H)*Pk => For H=I:
            # (I-K)*Pk => we do it manually: Pk - K*Pk
            # but simpler to do: Pk = (I - K*I) * Pk
            # We'll do: Pk = Pk - K@Pk
            # to remain in nopython, do it element-wise:

            # temp = K*Pk
            tmp00 = K00 * Pk[0, 0] + K01 * Pk[1, 0]
            tmp01 = K00 * Pk[0, 1] + K01 * Pk[1, 1]
            tmp10 = K10 * Pk[0, 0] + K11 * Pk[1, 0]
            tmp11 = K10 * Pk[0, 1] + K11 * Pk[1, 1]

            # Pk_new = Pk - temp
            Pk[0, 0] = Pk[0, 0] - tmp00
            Pk[0, 1] = Pk[0, 1] - tmp01
            Pk[1, 0] = Pk[1, 0] - tmp10
            Pk[1, 1] = Pk[1, 1] - tmp11

            # Update arrays
            covariances[i, 0, 0] = Pk[0, 0]
            covariances[i, 0, 1] = Pk[0, 1]
            covariances[i, 1, 0] = Pk[1, 0]
            covariances[i, 1, 1] = Pk[1, 1]

            particles[0, i] = Xupdt0
            particles[1, i] = Xupdt1

        # --- WEIGHTING & RESAMPLING ---
        if k == 0:
            # At the first step, compute weights based on distance to z
            for i in range(N):
                dx = particles[0, i] - z0
                dy = particles[1, i] - z1
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < 1e-6:
                    dist = 1e-6
                w[i] = 1.0 / dist
            # Normalize
            wsum = np.sum(w)
            if wsum > 1e-12:
                w /= wsum
            else:
                w[:] = 1.0 / float(N)

        # Effective sample size
        sum_w_sq = 0.0
        for i in range(N):
            sum_w_sq += w[i] * w[i]
        Neff = 1.0 / sum_w_sq
        Neff_array[k] = Neff

        # Systematic (or simple) resampling if Neff < some threshold
        if Neff < 300.0:
            # Build CDF
            cdf = np.empty(N, dtype=np.float64)
            cdf[0] = w[0]
            for i in range(1, N):
                cdf[i] = cdf[i - 1] + w[i]

            new_particles = np.zeros_like(particles)
            # Systematic resampling approach
            for j in range(N):
                uj = (np.random.rand() / N) + j / N
                # find cdf index
                idx = np.searchsorted(cdf, uj)
                if idx >= N:
                    idx = N - 1
                new_particles[0, j] = particles[0, idx]
                new_particles[1, j] = particles[1, idx]

            particles[:, :] = new_particles[:, :]
            # Reset uniform weights
            w[:] = 1.0 / float(N)

        # --- Compute corrected (mean) position from PF ---
        sumx = 0.0
        sumy = 0.0
        for i in range(N):
            sumx += particles[0, i] * w[i]
            sumy += particles[1, i] * w[i]
        Xcorr[0, k] = sumx
        Xcorr[1, k] = sumy

    # --- RMSE CALC ---
    # Compare Lat_tercom, Long_tercom to (ygps, xgps) or whichever mapping you have:
    # Adjust if your reference is xgps->lon, ygps->lat, etc.
    # Here we assume:
    #   "Lat_tercom" compares to ygps
    #   "Long_tercom" compares to xgps
    err_x = 0.0
    err_y = 0.0
    err_xpf = 0.0
    err_ypf = 0.0
    for k in range(50):
        dx_ter = ygps[k] - Lat_tercom[k]
        dy_ter = xgps[k] - Long_tercom[k]
        err_x += dx_ter * dx_ter
        err_y += dy_ter * dy_ter

        dx_pf = ygps[k] - Xcorr[0, k]
        dy_pf = xgps[k] - Xcorr[1, k]
        err_xpf += dx_pf * dx_pf
        err_ypf += dy_pf * dy_pf

    rmse_x = (err_x / 50.0) ** 0.5
    rmse_y = (err_y / 50.0) ** 0.5
    rmse_xpf = (err_xpf / 50.0) ** 0.5
    rmse_ypf = (err_ypf / 50.0) ** 0.5

    return rmse_x, rmse_y, rmse_xpf, rmse_ypf


# ----------------------------
# Example usage
if __name__ == "__main__":
    import time
    from scipy.io import loadmat

    # Load data (make sure everything is float64 and has correct shapes)
    measurements = loadmat('Measurements.mat')
    dem_heights = loadmat('DEM_heights.mat')
    dem_complete = loadmat('DEM_Complete.mat')
    data_v4 = loadmat('DataV4.mat')
    proc_noise = loadmat('ProcNoise2.mat')

    h_baro = measurements['h_baro'].flatten().astype(np.float64)
    h_radar = measurements['h_radar'].flatten().astype(np.float64)
    h_db = (h_baro[699:750] - h_radar[699:750]).astype(np.float64)

    dem_hgts = dem_heights['Z'].astype(np.float64)      # shape (rows, cols)
    dem_comp = dem_complete['M'].astype(np.float64)     # shape (rows, cols, 2)
    Data = data_v4['DataV4'].astype(np.float64)         # shape (>= 750, >= 9)
    pr_noise = proc_noise['P2'].astype(np.float64)      # shape (>=50, ?)

    # Extract xgps, ygps for the 699:750 range
    xgps = Data[699:750, 7].flatten()  # col 7 as x
    ygps = Data[699:750, 8].flatten()  # col 8 as y

    # Initialize filter parameters
    N = 1000
    particles = np.random.rand(2, N).astype(np.float64)
    covariances = np.zeros((N, 2, 2), dtype=np.float64)
    for i in range(N):
        covariances[i] = np.eye(2, dtype=np.float64)

    Qk = (np.eye(2) * 0.01).astype(np.float64)
    Rk = (np.eye(2) * 0.05).astype(np.float64)

    # Run filter
    start_time = time.time()
    rmse_x, rmse_y, rmse_xpf, rmse_ypf = particle_filter_numba(
        dem_hgts, dem_comp, h_db, pr_noise,
        particles, covariances, Qk, Rk, xgps, ygps, N
    )
    end_time = time.time()

    print(f"RMSE (x): {rmse_x}")
    print(f"RMSE (y): {rmse_y}")
    print(f"RMSE PF (x): {rmse_xpf}")
    print(f"RMSE PF (y): {rmse_ypf}")
    print("Elapsed time:", end_time - start_time, "seconds")

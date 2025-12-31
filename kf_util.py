import numpy as np

def kf_smooth_velocity(pred_vel, dt, p0,
                       q_pos=1e-4, q_vel=1e-3, r_vel=2.5e-1):
    """
    Simple linear Kalman filter that uses a constant-velocity model and
    TCN-predicted velocities as measurements.

    Args:
        pred_vel: (T, 2) numpy array of [vx, vy] from TCN.
        dt:       scalar time step (same units as your timestamps difference).
        p0:       (2,) initial position [px0, py0] (we typically use GT start).
        q_pos:    process noise variance on position (per dim).
        q_vel:    process noise variance on velocity (per dim).
        r_vel:    measurement noise variance on TCN velocity (per dim).

    Returns:
        pos_filt: (T, 2) filtered positions.
        vel_filt: (T, 2) filtered velocities.
    """
    T = pred_vel.shape[0]
    # State: [px, py, vx, vy]^T
    F = np.array([[1.0, 0.0, dt,  0.0],
                  [0.0, 1.0, 0.0, dt  ],
                  [0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0]], dtype=float)

    H = np.array([[0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0]], dtype=float)

    Q = np.diag([q_pos, q_pos, q_vel, q_vel])
    R = np.diag([r_vel, r_vel])

    # Initial state
    x = np.zeros((4, 1), dtype=float)
    x[0:2, 0] = p0  # start at ground-truth first position 
    P = np.eye(4, dtype=float) * 1.0

    pos_filt = np.zeros((T, 2), dtype=float)
    vel_filt = np.zeros((T, 2), dtype=float)

    for k in range(T):
        # Predict
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        # Measurement: TCN velocity
        z = pred_vel[k].reshape(2, 1)
        y = z - H @ x_pred            # innovation
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        # Update
        x = x_pred + K @ y
        P = (np.eye(4) - K @ H) @ P_pred

        pos_filt[k] = x[0:2, 0]
        vel_filt[k] = x[2:4, 0]

    return pos_filt, vel_filt

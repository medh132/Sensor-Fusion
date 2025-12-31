import torch
import torch.nn as nn


class DifferentiableKalmanFilter(nn.Module):
    """
    Differentiable Kalman Filter implementation in PyTorch
    Allows backpropagation through the filtering process
    """
    def __init__(self):
        super(DifferentiableKalmanFilter, self).__init__()
    
    def forward(self, pred_vel, dt, p0, q_pos, q_vel, r_vel):
        """
        Apply Kalman filter to velocity predictions
        
        Args:
            pred_vel: (batch, T, 2) predicted velocities [vx, vy]
            dt: scalar or (batch,) time step
            p0: (batch, 2) initial positions
            q_pos: scalar or tensor - process noise for position
            q_vel: scalar or tensor - process noise for velocity
            r_vel: scalar or tensor - measurement noise for velocity
            
        Returns:
            pos_filtered: (batch, T, 2) filtered positions
            vel_filtered: (batch, T, 2) filtered velocities
        """
        batch_size, T, _ = pred_vel.shape
        device = pred_vel.device
        
        # Convert dt to tensor if scalar
        if isinstance(dt, (int, float)):
            dt = torch.tensor(dt, device=device, dtype=pred_vel.dtype)
        
        # State transition matrix F (constant velocity model)
        # State: [px, py, vx, vy]
        F = torch.zeros(4, 4, device=device, dtype=pred_vel.dtype)
        F[0, 0] = 1.0
        F[1, 1] = 1.0
        F[2, 2] = 1.0
        F[3, 3] = 1.0
        F[0, 2] = dt
        F[1, 3] = dt
        
        # Measurement matrix H (we measure velocity only)
        H = torch.zeros(2, 4, device=device, dtype=pred_vel.dtype)
        H[0, 2] = 1.0
        H[1, 3] = 1.0
        
        # Process noise covariance Q
        Q = torch.zeros(4, 4, device=device, dtype=pred_vel.dtype)
        Q[0, 0] = q_pos
        Q[1, 1] = q_pos
        Q[2, 2] = q_vel
        Q[3, 3] = q_vel
        
        # Measurement noise covariance R
        R = torch.zeros(2, 2, device=device, dtype=pred_vel.dtype)
        R[0, 0] = r_vel
        R[1, 1] = r_vel
        
        # Initialize outputs
        pos_filtered = torch.zeros(batch_size, T, 2, device=device, dtype=pred_vel.dtype)
        vel_filtered = torch.zeros(batch_size, T, 2, device=device, dtype=pred_vel.dtype)
        
        # Process each sequence in batch
        for b in range(batch_size):
            # Initialize state: [px, py, vx, vy]
            x = torch.zeros(4, 1, device=device, dtype=pred_vel.dtype)
            x[0:2, 0] = p0[b]
            
            # Initialize covariance
            P = torch.eye(4, device=device, dtype=pred_vel.dtype) * 1.0
            
            for t in range(T):
                # Predict step
                x_pred = F @ x
                P_pred = F @ P @ F.T + Q
                
                # Measurement
                z = pred_vel[b, t].unsqueeze(1)  # (2, 1)
                
                # Innovation
                y = z - H @ x_pred
                S = H @ P_pred @ H.T + R
                
                # Kalman gain
                # Add small value to diagonal for numerical stability
                S_reg = S + torch.eye(2, device=device, dtype=pred_vel.dtype) * 1e-6
                K = P_pred @ H.T @ torch.inverse(S_reg)
                
                # Update step
                x = x_pred + K @ y
                I_KH = torch.eye(4, device=device, dtype=pred_vel.dtype) - K @ H
                P = I_KH @ P_pred
                
                # Store filtered results
                pos_filtered[b, t] = x[0:2, 0]
                vel_filtered[b, t] = x[2:4, 0]
        
        return pos_filtered, vel_filtered


class BatchDifferentiableKF(nn.Module):
    """
    Vectorized version for better performance (processes whole batch at once)
    More complex but faster
    """
    def __init__(self):
        super(BatchDifferentiableKF, self).__init__()
    
    def forward(self, pred_vel, dt, p0, q_pos, q_vel, r_vel):
        """
        Batched Kalman filter implementation
        
        Args:
            pred_vel: (batch, T, 2) predicted velocities
            dt: scalar time step
            p0: (batch, 2) initial positions
            q_pos, q_vel, r_vel: noise parameters
            
        Returns:
            pos_filtered: (batch, T, 2) filtered positions
            vel_filtered: (batch, T, 2) filtered velocities
        """
        batch_size, T, _ = pred_vel.shape
        device = pred_vel.device
        dtype = pred_vel.dtype
        
        if isinstance(dt, (int, float)):
            dt = torch.tensor(dt, device=device, dtype=dtype)
        
        # Build state transition matrix F for batch
        #F = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        F = torch.eye(4, device=device).expand(batch_size, 4, 4).clone()
        F[:, 0, 2] = dt
        F[:, 1, 3] = dt
        
        # Measurement matrix H
        H = torch.zeros(batch_size, 2, 4, device=device, dtype=dtype)
        H[:, 0, 2] = 1.0
        H[:, 1, 3] = 1.0
        
        # Process noise Q
        Q = torch.zeros(batch_size, 4, 4, device=device, dtype=dtype)
        Q[:, 0, 0] = q_pos
        Q[:, 1, 1] = q_pos
        Q[:, 2, 2] = q_vel
        Q[:, 3, 3] = q_vel
        
        # Measurement noise R
        R = torch.zeros(batch_size, 2, 2, device=device, dtype=dtype)
        R[:, 0, 0] = r_vel
        R[:, 1, 1] = r_vel
        
        # Initialize state
        x = torch.zeros(batch_size, 4, 1, device=device, dtype=dtype)
        x[:, 0:2, 0] = p0
        
        # Initialize covariance
        P = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1).clone()
        
        pos_filtered = []
        vel_filtered = []
        
        for t in range(T):
            # Predict
            x_pred = torch.bmm(F, x)  # (batch, 4, 1)
            P_pred = torch.bmm(torch.bmm(F, P), F.transpose(1, 2)) + Q
            
            # Measurement
            z = pred_vel[:, t].unsqueeze(2)  # (batch, 2, 1)
            
            # Innovation
            y = z - torch.bmm(H, x_pred)
            S = torch.bmm(torch.bmm(H, P_pred), H.transpose(1, 2)) + R
            
            # Numerical stability
            #eye_2 = torch.eye(2, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
            eye_2 = torch.eye(2, device=device, dtype=dtype).expand(batch_size, 2, 2).clone()
            S_reg = S + eye_2 * 1e-6
            
            # Kalman gain
            K = torch.bmm(torch.bmm(P_pred, H.transpose(1, 2)), torch.inverse(S_reg))
            
            # Update
            x = x_pred + torch.bmm(K, y)
            #eye_4 = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
            eye_4 = torch.eye(4, device=device, dtype=dtype).expand(batch_size, 4, 4).clone()
            I_KH = eye_4 - torch.bmm(K, H)
            P = torch.bmm(I_KH, P_pred)
            
            # Store results
            pos_filtered.append(x[:, 0:2, 0])
            vel_filtered.append(x[:, 2:4, 0])
        
        pos_filtered = torch.stack(pos_filtered, dim=1)  # (batch, T, 2)
        vel_filtered = torch.stack(vel_filtered, dim=1)  # (batch, T, 2)
        
        return pos_filtered, vel_filtered


def integrate_velocities(velocities, dt, p0):
    """
    Simple integration of velocities to positions
    
    Args:
        velocities: (batch, T, 2)
        dt: scalar
        p0: (batch, 2) initial position
        
    Returns:
        positions: (batch, T, 2)
    """
    # Multiply by dt and cumsum
    displacements = velocities * dt
    positions = torch.cumsum(displacements, dim=1)
    
    # Add initial position
    positions = positions + p0.unsqueeze(1)
    
    return positions

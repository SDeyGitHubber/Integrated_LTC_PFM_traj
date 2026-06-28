"""
Trajectory Linearity Classification Utility

Classifies agent trajectories as LINEAR or NON-LINEAR based on 
regression line fitting error. This enables targeted training 
on different motion patterns.

Methodology:
-----------
1. Fit linear regression line to history + ground truth trajectory
2. Calculate mean squared error (MSE) between line and points
3. Use adaptive threshold (mean + k*std) to classify trajectories
4. Linear: low deviation from straight-line motion
5. Non-linear: high deviation (turns, curves, stops)
"""

import torch
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class TrajectoryLinearityClassifier:
    """
    Classifies trajectories as linear or non-linear based on regression fitting.
    """
    
    def __init__(self, threshold_method='adaptive', threshold_value=None, k_std=1.0):
        """
        Initialize the classifier.
        
        Args:
            threshold_method (str): 'adaptive' (auto-compute from data) or 'fixed'
            threshold_value (float): Fixed threshold for MSE (if method='fixed')
            k_std (float): Number of standard deviations for adaptive threshold
                          Higher k = more trajectories classified as linear
                          Recommended: 0.5 (strict), 1.0 (balanced), 1.5 (lenient)
        """
        self.threshold_method = threshold_method
        self.threshold_value = threshold_value
        self.k_std = k_std
        self.computed_threshold = None
        self.mse_stats = None
    
    def fit_linear_regression(self, trajectory):
        """
        Fit linear regression line to a single trajectory.
        
        Args:
            trajectory: numpy array (T, 2) - sequence of (x, y) positions
            
        Returns:
            mse: Mean squared error between trajectory and fitted line
            line_params: (slope, intercept) of fitted line
            fitted_line: numpy array (T, 2) - points on fitted line
        """
        T = len(trajectory)
        
        # Use time indices as x for regression
        t = np.arange(T).reshape(-1, 1)
        
        # Fit separate regressions for x and y coordinates
        reg_x = LinearRegression().fit(t, trajectory[:, 0])
        reg_y = LinearRegression().fit(t, trajectory[:, 1])
        
        # Predict positions along fitted lines
        pred_x = reg_x.predict(t)
        pred_y = reg_y.predict(t)
        fitted_line = np.column_stack([pred_x, pred_y])
        
        # Calculate MSE
        mse = mean_squared_error(trajectory, fitted_line)
        
        # Store line parameters
        line_params = {
            'x_slope': reg_x.coef_[0],
            'x_intercept': reg_x.intercept_,
            'y_slope': reg_y.coef_[0],
            'y_intercept': reg_y.intercept_
        }
        
        return mse, line_params, fitted_line
    
    def compute_adaptive_threshold(self, mse_values):
        """
        Compute adaptive threshold based on MSE distribution.
        
        Uses: threshold = mean + k * std
        
        This ensures that the threshold adapts to the dataset's 
        natural distribution of trajectory linearity.
        
        Args:
            mse_values: list or array of MSE values
            
        Returns:
            threshold: Computed threshold value
        """
        mse_array = np.array(mse_values)
        mean_mse = np.mean(mse_array)
        std_mse = np.std(mse_array)
        
        threshold = mean_mse + self.k_std * std_mse
        
        self.mse_stats = {
            'mean': mean_mse,
            'std': std_mse,
            'min': np.min(mse_array),
            'max': np.max(mse_array),
            'median': np.median(mse_array),
            'q25': np.percentile(mse_array, 25),
            'q75': np.percentile(mse_array, 75)
        }
        
        self.computed_threshold = threshold
        return threshold
    
    def classify_batch(self, history, future):
        """
        Classify a batch of trajectories as linear or non-linear.
        
        Args:
            history: Tensor (B, A, H, 2) - history positions
            future: Tensor (B, A, T, 2) - future positions (ground truth)
            
        Returns:
            classifications: Tensor (B, A) - 1 for linear, 0 for non-linear
            mse_values: Tensor (B, A) - MSE values for each trajectory
            regression_lines: Tensor (B, A, H+T, 2) - fitted lines
            threshold: float - threshold used for classification
        """
        B, A, H, _ = history.shape
        _, _, T, _ = future.shape
        
        # Convert to numpy
        history_np = history.cpu().numpy()
        future_np = future.cpu().numpy()
        
        # Concatenate history and future
        full_trajectory = np.concatenate([history_np, future_np], axis=2)  # (B, A, H+T, 2)
        
        # Storage
        mse_values = np.zeros((B, A))
        regression_lines = np.zeros((B, A, H + T, 2))
        
        # Fit regression for each trajectory
        for b in range(B):
            for a in range(A):
                traj = full_trajectory[b, a]
                mse, _, fitted_line = self.fit_linear_regression(traj)
                mse_values[b, a] = mse
                regression_lines[b, a] = fitted_line
        
        # Determine threshold
        if self.threshold_method == 'adaptive':
            threshold = self.compute_adaptive_threshold(mse_values.flatten())
        else:
            threshold = self.threshold_value
        
        # Classify: 1 if linear (MSE < threshold), 0 if non-linear
        classifications = (mse_values < threshold).astype(np.float32)
        
        # Convert back to tensors
        classifications = torch.from_numpy(classifications).to(history.device)
        mse_values = torch.from_numpy(mse_values).to(history.device)
        regression_lines = torch.from_numpy(regression_lines).to(history.device)
        
        return classifications, mse_values, regression_lines, threshold
    
    def get_segment_masks(self, classifications):
        """
        Get boolean masks for linear and non-linear segments.
        
        Args:
            classifications: Tensor (B, A) - 1 for linear, 0 for non-linear
            
        Returns:
            linear_mask: Tensor (B, A) - True for linear trajectories
            nonlinear_mask: Tensor (B, A) - True for non-linear trajectories
        """
        linear_mask = classifications == 1
        nonlinear_mask = classifications == 0
        
        return linear_mask, nonlinear_mask
    
    def print_statistics(self):
        """Print statistics about MSE distribution and threshold."""
        if self.mse_stats is None:
            print("No statistics available. Run classify_batch first.")
            return
        
        print("="*80)
        print("TRAJECTORY LINEARITY STATISTICS")
        print("="*80)
        print(f"MSE Distribution:")
        print(f"  Mean:       {self.mse_stats['mean']:.6f}")
        print(f"  Std Dev:    {self.mse_stats['std']:.6f}")
        print(f"  Min:        {self.mse_stats['min']:.6f}")
        print(f"  Max:        {self.mse_stats['max']:.6f}")
        print(f"  Median:     {self.mse_stats['median']:.6f}")
        print(f"  Q25:        {self.mse_stats['q25']:.6f}")
        print(f"  Q75:        {self.mse_stats['q75']:.6f}")
        print(f"\nThreshold: {self.computed_threshold:.6f}")
        print(f"Method: {self.threshold_method} (k={self.k_std})")
        print("="*80)


def split_batch_by_linearity(batch_data, classifications):
    """
    Split a batch into linear and non-linear segments.
    
    Args:
        batch_data: Tuple of tensors (history, future, ...)
        classifications: Tensor (B, A) - linearity classifications
        
    Returns:
        linear_batch: Tuple of tensors for linear trajectories
        nonlinear_batch: Tuple of tensors for non-linear trajectories
        linear_indices: List of (batch_idx, agent_idx) for linear trajectories
        nonlinear_indices: List of (batch_idx, agent_idx) for non-linear trajectories
    """
    B, A = classifications.shape
    
    # Get indices
    linear_mask = classifications == 1
    nonlinear_mask = classifications == 0
    
    linear_indices = torch.nonzero(linear_mask, as_tuple=False).cpu().numpy()
    nonlinear_indices = torch.nonzero(nonlinear_mask, as_tuple=False).cpu().numpy()
    
    # Extract trajectories
    linear_data = []
    nonlinear_data = []
    
    for tensor in batch_data:
        if tensor is not None and isinstance(tensor, torch.Tensor):
            if len(tensor.shape) >= 2 and tensor.shape[0] == B and tensor.shape[1] == A:
                # Extract linear trajectories
                linear_tensor = tensor[linear_mask]
                nonlinear_tensor = tensor[nonlinear_mask]
                
                linear_data.append(linear_tensor)
                nonlinear_data.append(nonlinear_tensor)
            else:
                # Keep as-is if not (B, A, ...) shape
                linear_data.append(tensor)
                nonlinear_data.append(tensor)
        else:
            linear_data.append(tensor)
            nonlinear_data.append(tensor)
    
    return tuple(linear_data), tuple(nonlinear_data), linear_indices, nonlinear_indices


def visualize_linearity_classification(history, future, classifications, mse_values, 
                                       regression_lines, save_path=None, num_samples=4):
    """
    Visualize linearity classification results.
    
    Args:
        history: Tensor (B, A, H, 2)
        future: Tensor (B, A, T, 2)
        classifications: Tensor (B, A)
        mse_values: Tensor (B, A)
        regression_lines: Tensor (B, A, H+T, 2)
        save_path: Path to save figure (optional)
        num_samples: Number of samples to visualize
    """
    import matplotlib.pyplot as plt
    
    B, A, H, _ = history.shape
    _, _, T, _ = future.shape
    
    # Convert to numpy
    history_np = history.cpu().numpy()
    future_np = future.cpu().numpy()
    classifications_np = classifications.cpu().numpy()
    mse_values_np = mse_values.cpu().numpy()
    regression_lines_np = regression_lines.cpu().numpy()
    
    # Select samples
    num_samples = min(num_samples, B * A)
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    sample_idx = 0
    for b in range(B):
        for a in range(A):
            if sample_idx >= num_samples:
                break
            
            ax = axes[sample_idx]
            
            # Plot history
            ax.plot(history_np[b, a, :, 0], history_np[b, a, :, 1], 
                   'o-', color='blue', label='History', markersize=4)
            
            # Plot future
            ax.plot(future_np[b, a, :, 0], future_np[b, a, :, 1], 
                   'o-', color='green', label='Ground Truth', markersize=4)
            
            # Plot regression line
            ax.plot(regression_lines_np[b, a, :, 0], regression_lines_np[b, a, :, 1], 
                   '--', color='red', label='Regression Line', linewidth=2)
            
            # Title with classification
            is_linear = classifications_np[b, a] == 1
            mse = mse_values_np[b, a]
            label = "LINEAR" if is_linear else "NON-LINEAR"
            color = "green" if is_linear else "red"
            
            ax.set_title(f"{label} | MSE: {mse:.4f}", color=color, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            
            sample_idx += 1
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()

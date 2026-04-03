import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from bubbleformer.utils.sdf_reinit import sdf_reinit_fast_marching, sdf_reinit_sussman, sdf_reinit_drift, verify_sdf
import torch

if __name__ == "__main__":
    # Create a circle
    n = 200
    x = np.linspace(-2, 2, n)
    y = np.linspace(-2, 2, n)
    X, Y = np.meshgrid(x, y)
    
    # Initial level set: circle of radius 0.5
    radius = 0.5
    phi_init = -(np.sqrt(X**2 + Y**2) - radius).astype(np.float32)
    
    phi_error = phi_init.copy()
    phi_error[0:30, 0:30] += np.random.normal(0, 0.1, size=(30, 30))
    print(phi_error)
    
    # Compute signed distance function
    dx = x[1] - x[0]
    #phi_sdf_fast = sdf_reinit_fast_marching(torch.from_numpy(phi_error), dx, far_threshold=-0.4).numpy()
    phi_sdf = sdf_reinit_sussman(torch.from_numpy(phi_error), dx, dx, 100).numpy()

    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5), layout="constrained")
    
    # Initial level set
    c1 = ax1.contourf(X, Y, phi_init, levels=20, cmap='RdBu_r', norm=TwoSlopeNorm(vcenter=0))
    ax1.contour(X, Y, phi_init, levels=[0], colors='black', linewidths=2)
    ax1.set_title('Initial Level Set')
    ax1.set_aspect('equal')
    plt.colorbar(c1, ax=ax1, fraction=0.04, pad=0.05)
    
    # Initial level set with error
    c2 = ax2.contourf(X, Y, phi_error, levels=20, cmap='RdBu_r', norm=TwoSlopeNorm(vcenter=0))
    ax2.contour(X, Y, phi_error, levels=[0], colors='black', linewidths=2)
    ax2.set_title('Initial Level Set with Error')
    ax2.set_aspect('equal')
    plt.colorbar(c2, ax=ax2, fraction=0.04, pad=0.05)
    
    # Corrected signed distance function
    c3 = ax3.contourf(X, Y, phi_sdf, levels=20, cmap='RdBu_r', norm=TwoSlopeNorm(vcenter=0))
    ax3.contour(X, Y, phi_sdf, levels=[0], colors='black', linewidths=2)
    ax3.set_title('Corrected Signed Distance Function')
    ax3.set_aspect('equal')
    plt.colorbar(c3, ax=ax3, fraction=0.04, pad=0.05)
    
    plt.savefig("sdf_reinit_test.png")
    
    init_mean, init_std = verify_sdf(torch.from_numpy(phi_init), dx.item())
    error_mean, error_std = verify_sdf(torch.from_numpy(phi_error), dx.item())
    sdf_mean, sdf_std = verify_sdf(torch.from_numpy(phi_sdf), dx)
    print(f"Initial SDF Mean: {init_mean:.4f}, Std: {init_std:.4f}")
    print(f"Error SDF Mean: {error_mean:.4f}, Std: {error_std:.4f}")
    print(f"SDF SDF Mean: {sdf_mean:.4f}, Std: {sdf_std:.4f}")
    
    drift = sdf_reinit_drift(torch.from_numpy(phi_error), torch.from_numpy(phi_sdf), dx)
    print(f"SDF Reinit Drift (should be less than {dx}): {drift:.4f}")
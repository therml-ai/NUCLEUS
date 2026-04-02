import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from nucleus.utils.sdf_reinit import sdf_reinit
import torch

def verify(phi_sdf, dx):
    # Verify it's a distance function: check |∇phi| ≈ 1
    grad_y, grad_x = np.gradient(phi_sdf, dx)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    print(f"Gradient magnitude mean: {grad_magnitude.mean():.4f}")
    print(f"Gradient magnitude std: {grad_magnitude.std():.4f}")

# Example usage
if __name__ == "__main__":
    # Create a circle
    n = 200
    x = np.linspace(-2, 2, n)
    y = np.linspace(-2, 2, n)
    X, Y = np.meshgrid(x, y)
    
    # Initial level set: circle of radius 0.5
    radius = 0.5
    phi_init =  - (np.sqrt(X**2 + Y**2) - radius)
    
    phi_error = phi_init.copy()
    #phi_error[phi_error > 0] = phi_error[phi_error > 0] + 0.1
    phi_error[0:30, 0:30] = -0.5
    #phi_error[30:40, 30:40] = -0.1
    print(phi_error)
    
    #torch.nn.functional.interpolate(phi_error, size=(100, 100), mode="bicubic")
    
    # Compute signed distance function
    dx = x[1] - x[0]
    phi_sdf = sdf_reinit(torch.from_numpy(phi_error), dx, far_threshold=-0.4).numpy()

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
    
    verify(phi_init, dx)
    verify(phi_error, dx)
    verify(phi_sdf, dx)
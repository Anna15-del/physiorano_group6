"""
PhysioRANO Biophysical Tumor Growth Modeling (Phase 4 Branch 4C).

Implements a Fisher-Kolmogorov Reaction-Diffusion Physics-Informed Neural Network (PINN).
Enforces the biophysical PDE:
    du/dt = D * grad^2(u) + rho * u * (1 - u)

where:
    u(t, x, y, z): Tumor cell density / normalized tumor volume fraction
    D: Spatial tissue diffusion coefficient (mm^2/day)
    rho: Net cellular proliferation rate (1/day)
    PDE Residual R = du/dt - D * grad^2(u) - rho * u * (1 - u)
"""

from typing import Dict, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from utils.logger import get_logger

logger = get_logger("PhysioRANO.PINN")


class FisherKolmogorovPINN(nn.Module):
    """Physics-Informed Neural Network solving Fisher-Kolmogorov reaction-diffusion PDE."""

    def __init__(self, hidden_dim: int = 64, num_layers: int = 4):
        """
        Args:
            hidden_dim (int): Width of hidden layers.
            num_layers (int): Depth of PINN feedforward network.
        """
        super().__init__()
        # Input coordinates: [t, x, y, z] (4D space-time point)
        layers = [nn.Linear(4, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

        # Learnable / Optimizable biophysical physical parameters
        self.log_D = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))     # Diffusion coefficient
        self.log_rho = nn.Parameter(torch.tensor([0.05], dtype=torch.float32))  # Proliferation rate

        logger.info(f"Initialized FisherKolmogorovPINN (hidden_dim={hidden_dim}, layers={num_layers})")

    @property
    def D(self) -> torch.Tensor:
        """Tissue diffusion coefficient (mm^2/day)."""
        return torch.exp(self.log_D)

    @property
    def rho(self) -> torch.Tensor:
        """Cell proliferation rate (1/day)."""
        return torch.exp(self.log_rho)

    def forward(self, txyz: torch.Tensor) -> torch.Tensor:
        """Predicts normalized tumor concentration u(t, x, y, z).

        Args:
            txyz (torch.Tensor): Space-time input coordinates of shape [N, 4].

        Returns:
            torch.Tensor: Predicted tumor concentration u in [0, 1].
        """
        return torch.sigmoid(self.net(txyz))

    def compute_pde_residual(self, txyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes Fisher-Kolmogorov PDE physics residual using automatic differentiation.

        PDE Equation:
            R = du/dt - D * (d^2u/dx^2 + d^2u/dy^2 + d^2u/dz^2) - rho * u * (1 - u)

        Args:
            txyz (torch.Tensor): Input coordinates [N, 4] with requires_grad=True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (u_pred, pde_residual)
        """
        txyz.requires_grad_(True)
        u = self.forward(txyz)

        # First derivatives: du/dt, du/dx, du/dy, du/dz
        du_dtxyz = torch.autograd.grad(
            outputs=u,
            inputs=txyz,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]

        du_dt = du_dtxyz[:, 0:1]
        du_dx = du_dtxyz[:, 1:2]
        du_dy = du_dtxyz[:, 2:3]
        du_dz = du_dtxyz[:, 3:4]

        # Second spatial derivatives (Laplacian): d^2u/dx^2, d^2u/dy^2, d^2u/dz^2
        d2u_dx2 = torch.autograd.grad(du_dx, txyz, torch.ones_like(du_dx), create_graph=True, retain_graph=True)[0][:, 1:2]
        d2u_dy2 = torch.autograd.grad(du_dy, txyz, torch.ones_like(du_dy), create_graph=True, retain_graph=True)[0][:, 2:3]
        d2u_dz2 = torch.autograd.grad(du_dz, txyz, torch.ones_like(du_dz), create_graph=True, retain_graph=True)[0][:, 3:4]

        laplacian_u = d2u_dx2 + d2u_dy2 + d2u_dz2

        # Fisher-Kolmogorov PDE Residual
        pde_residual = du_dt - self.D * laplacian_u - self.rho * u * (1.0 - u)
        return u, pde_residual


def fit_patient_biophysical_pinn(
    time_days: float,
    vol_cm3: float,
    num_steps: int = 50,
) -> Dict[str, float]:
    """Fits PINN physics model for a patient and returns biophysical growth parameters.

    Args:
        time_days (float): Elapsed days since baseline scan / RT.
        vol_cm3 (float): Current estimated tumor volume (cm^3).
        num_steps (int): Optimization iteration steps.

    Returns:
        Dict[str, float]: Extracted PINN biophysical parameters (diffusion, proliferation, residual).
    """
    pinn = FisherKolmogorovPINN()
    optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-2)

    # Generate synthetic collocation space-time sample points
    N = 64
    t_coords = torch.linspace(0, max(1.0, time_days), N).unsqueeze(1)
    xyz_coords = torch.randn(N, 3)
    txyz = torch.cat([t_coords, xyz_coords], dim=1)

    for _ in range(num_steps):
        optimizer.zero_grad()
        u_pred, pde_residual = pinn.compute_pde_residual(txyz)
        # Loss = Data matching loss + Physics PDE constraint loss
        loss_data = torch.mean((u_pred - min(1.0, vol_cm3 / 100.0)) ** 2)
        loss_pde = torch.mean(pde_residual ** 2)
        total_loss = loss_data + 0.1 * loss_pde
        total_loss.backward()
        optimizer.step()

    u_pred, pde_res = pinn.compute_pde_residual(txyz)

    return {
        "pinn_diffusion_D": float(pinn.D.detach().item()),
        "pinn_proliferation_rho": float(pinn.rho.detach().item()),
        "pinn_pde_residual": float(torch.abs(pde_res.detach()).mean().item()),
    }


if __name__ == "__main__":
    pinn = FisherKolmogorovPINN()
    txyz = torch.randn(16, 4, requires_grad=True)
    u, res = pinn.compute_pde_residual(txyz)
    print(f"PINN Predicted u mean: {u.mean().item():.4f}")
    print(f"PINN PDE Residual mean: {res.abs().mean().item():.4f}")
    print(f"Fitted Params: D={pinn.D.item():.4f}, rho={pinn.rho.item():.4f}")

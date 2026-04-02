from typing import List, Union
import torch
import torch.nn as nn

def eikonal_loss(phi):
    """
    This enforces the eikonal equation: ||grad(phi)|| = 1.
    Args:
        phi: SDF torch.Tensor (B, T, H, W).
    """
    dx = 1/32
    grad_phi_y, grad_phi_x = torch.gradient(phi, spacing=dx, dim=(-2, -1), edge_order=1)
    grad_mag = torch.sqrt(grad_phi_y**2 + grad_phi_x**2)
    eikonal_mse = (grad_mag - 1.0) ** 2
    return eikonal_mse.mean()

class LpLoss(nn.Module):
    """
    Lp loss on a tensor (b, n1, n2, ..., nd)
    Args:
        d (int): Number of dimensions to flatten from right
        p (int): Power of the norm
        reduce_dims (List[int]): Dimensions to reduce
        reductions (List[str]): Reductions to apply
    """
    def __init__(
            self,
            d: int = 1,
            p: int = 2,
            reduce_dims: Union[int, List[int]] = 0,
            reductions: Union[str, List[str]] = "sum"
        ):
        super().__init__()

        self.d = d
        self.p = p

        if isinstance(reduce_dims, int):
            self.reduce_dims = [reduce_dims]
        else:
            self.reduce_dims = reduce_dims

        if self.reduce_dims is not None:
            if isinstance(reductions, str):
                assert reductions == "sum" or reductions == "mean"
                self.reductions = [reductions] * len(self.reduce_dims)
            else:
                for reduction in reductions:
                    assert reduction == "sum" or reduction == "mean"
                self.reductions = reductions

    def reduce_all(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reduce the tensor along the specified dimensions
        Args:
            x (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: Reduced tensor
        """
        for j, reduce_dim in enumerate(self.reduce_dims):
            if self.reductions[j] == "sum":
                x = torch.sum(x, dim=reduce_dim, keepdim=True)
            else:
                x = torch.mean(x, dim=reduce_dim, keepdim=True)
        return x

    def forward(
            self,
            y_pred: torch.Tensor,
            y: torch.Tensor
        ) -> torch.Tensor:
        """
        Args:
            y_pred (torch.Tensor): Predicted tensor
            y (torch.Tensor): Target tensor
        Returns:
            torch.Tensor: Lp loss
        """
        diff = torch.norm(
            torch.flatten(y_pred, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d),
            p=self.p,
            dim=-1,
            keepdim=False,
        )
        ynorm = torch.norm(
            torch.flatten(y, start_dim=-self.d), p=self.p, dim=-1, keepdim=False
        )

        diff = diff / ynorm

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()

        return diff

class L1Loss(nn.Module):
    def __init__(self, scales: List[float]):
        super().__init__()
        self.loss = nn.L1Loss()
        self.scales = scales
    
    def forward(self, pred, target, bulk_temp: torch.Tensor):
        # compute loss with temperatures mapped to bulk temp 0 for easier training.
        pred_temp = pred[:, :, 1, :, :] - bulk_temp[:, None, None, None]
        target_temp = target[:, :, 1, :, :] - bulk_temp[:, None, None, None]
        pred_bulk = torch.stack([pred[:, :, 0, :, :], pred_temp, pred[:, :, 2, :, :], pred[:, :, 3, :, :]], dim=2)
        target_bulk = torch.stack([target[:, :, 0, :, :], target_temp, target[:, :, 2, :, :], target[:, :, 3, :, :]], dim=2)
        return self.loss(pred_bulk, target_bulk)
        
        
class L1RelativeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss(reduction="none")

    def forward(self, pred, target, bulk_temp: torch.Tensor):
        eps = torch.full_like(target[:, :, 0, :, :], 1e-4)
        
        sdf_norm = torch.norm(target[:, :, 0, :, :], p=1, dim=(-3, -2, -1))
        velx_norm = torch.norm(target[:, :, 2, :, :], p=1, dim=(-3, -2, -1))
        vely_norm = torch.norm(target[:, :, 3, :, :], p=1, dim=(-3, -2, -1))
        temp_norm = torch.norm(target[:, :, 1, :, :] - bulk_temp[:, None, None, None], p=1, dim=(-3, -2, -1))

        # The range of values are quite large, so to make the losses a little closer to the
        # non-relative loss, we divide by the max of the norms.
        norm_denom = torch.max(
            torch.stack([sdf_norm, velx_norm, vely_norm, temp_norm], dim=0), dim=0
        ).values
        
        sdf_loss = self.loss(pred[:, :, 0, :, :], target[:, :, 0, :, :]) / (sdf_norm / norm_denom)[:, None, None, None]
        velx_loss = self.loss(pred[:, :, 2, :, :], target[:, :, 2, :, :]) / (velx_norm / norm_denom)[:, None, None, None]
        vely_loss = self.loss(pred[:, :, 3, :, :], target[:, :, 3, :, :]) / (vely_norm / norm_denom)[:, None, None, None]

        pred_temp = pred[:, :, 1, :, :]
        target_temp = target[:, :, 1, :, :]
        temp_loss = self.loss(pred_temp, target_temp) / (temp_norm / norm_denom)[:, None, None, None]
        
        # Add each loss and take mean over batch dimensions.
        return (sdf_loss + temp_loss + velx_loss + vely_loss).mean()
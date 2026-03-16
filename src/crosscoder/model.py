from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config


class SPARCCrossCoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        expansion_factor: int,
        topk: int,
        forced_shared_fraction: float = config.FORCED_SHARED_FRACTION,
        seed: int = config.SEED,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.dict_size = input_dim * expansion_factor
        self.topk = topk
        self.forced_shared_fraction = forced_shared_fraction
        
        torch.manual_seed(seed)
        
        self.encoder_u = nn.Linear(input_dim, self.dict_size)
        self.encoder_c = nn.Linear(input_dim, self.dict_size)
        
        self.decoder_u = nn.Linear(self.dict_size, input_dim, bias=False)
        self.decoder_c = nn.Linear(self.dict_size, input_dim, bias=False)
        
        n_forced_shared = int(self.dict_size * forced_shared_fraction)
        self.register_buffer(
            "forced_shared_indices",
            torch.randperm(self.dict_size)[:n_forced_shared]
        )
        self.n_forced_shared = n_forced_shared
        
        self._init_weights()
        self._init_forced_shared()
    
    def _init_weights(self):
        for module in [self.encoder_u, self.encoder_c]:
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            nn.init.zeros_(module.bias)
        
        for module in [self.decoder_u, self.decoder_c]:
            nn.init.kaiming_uniform_(module.weight, nonlinearity='linear')
    
    def _init_forced_shared(self):
        with torch.no_grad():
            self.decoder_c.weight.data[:, self.forced_shared_indices] = \
                self.decoder_u.weight.data[:, self.forced_shared_indices].clone()
    
    def global_topk_balanced(self, h_u: torch.Tensor, h_c: torch.Tensor) -> torch.Tensor:
        batch_size = h_u.shape[0]
        k_half = self.topk // 2
        
        _, topk_u_indices = torch.topk(h_u, k_half, dim=-1)
        _, topk_c_indices = torch.topk(h_c, k_half, dim=-1)
        
        mask = torch.zeros_like(h_u, dtype=torch.bool)
        mask.scatter_(-1, topk_u_indices, True)
        mask.scatter_(-1, topk_c_indices, True)
        
        return mask
    
    def encode(self, x_u: torch.Tensor, x_c: torch.Tensor):
        h_u = self.encoder_u(x_u)
        h_c = self.encoder_c(x_c)
        
        mask = self.global_topk_balanced(h_u, h_c)
        
        z_u = F.relu(h_u) * mask.float()
        z_c = F.relu(h_c) * mask.float()
        
        return z_u, z_c, mask
    
    def decode(self, z_u: torch.Tensor, z_c: torch.Tensor):
        x_u_hat = self.decoder_u(z_u)
        x_c_hat = self.decoder_c(z_c)
        return x_u_hat, x_c_hat
    
    def cross_decode(self, z_u: torch.Tensor, z_c: torch.Tensor):
        x_u_from_c = self.decoder_u(z_c)
        x_c_from_u = self.decoder_c(z_u)
        return x_u_from_c, x_c_from_u
    
    def forward(self, x_u: torch.Tensor, x_c: torch.Tensor):
        z_u, z_c, mask = self.encode(x_u, x_c)
        x_u_hat, x_c_hat = self.decode(z_u, z_c)
        x_u_cross, x_c_cross = self.cross_decode(z_u, z_c)
        
        return {
            "z_u": z_u,
            "z_c": z_c,
            "mask": mask,
            "x_u_hat": x_u_hat,
            "x_c_hat": x_c_hat,
            "x_u_cross": x_u_cross,
            "x_c_cross": x_c_cross,
        }
    
    def compute_loss(
        self,
        x_u: torch.Tensor,
        x_c: torch.Tensor,
        outputs: dict,
        lambda_sparsity: float = config.LAMBDA_SPARSITY,
        lambda_cross: float = config.LAMBDA_CROSS,
        lambda_shared_multiplier: float = config.LAMBDA_SHARED_MULTIPLIER,
    ) -> dict:
        z_u = outputs["z_u"]
        z_c = outputs["z_c"]
        x_u_hat = outputs["x_u_hat"]
        x_c_hat = outputs["x_c_hat"]
        x_u_cross = outputs["x_u_cross"]
        x_c_cross = outputs["x_c_cross"]
        
        loss_recon_u = F.mse_loss(x_u, x_u_hat)
        loss_recon_c = F.mse_loss(x_c, x_c_hat)
        loss_self = loss_recon_u + loss_recon_c
        
        loss_cross_u = F.mse_loss(x_u, x_u_cross)
        loss_cross_c = F.mse_loss(x_c, x_c_cross)
        loss_cross = loss_cross_u + loss_cross_c
        
        W_u_norms = self.decoder_u.weight.norm(dim=0)
        W_c_norms = self.decoder_c.weight.norm(dim=0)
        decoder_norm_sum = W_u_norms + W_c_norms
        
        z_combined = (z_u.abs() + z_c.abs()) / 2
        
        forced_mask = torch.zeros(self.dict_size, device=x_u.device, dtype=torch.bool)
        forced_mask[self.forced_shared_indices] = True
        
        z_forced = z_combined[:, forced_mask]
        decoder_norms_forced = decoder_norm_sum[forced_mask]
        loss_l1_forced = (z_forced * decoder_norms_forced.unsqueeze(0)).sum(dim=-1).mean()
        
        z_standard = z_combined[:, ~forced_mask]
        decoder_norms_standard = decoder_norm_sum[~forced_mask]
        loss_l1_standard = (z_standard * decoder_norms_standard.unsqueeze(0)).sum(dim=-1).mean()
        
        lambda_shared = lambda_sparsity * lambda_shared_multiplier
        loss_sparsity = lambda_shared * loss_l1_forced + lambda_sparsity * loss_l1_standard
        
        total_loss = loss_self + lambda_cross * loss_cross + loss_sparsity
        
        return {
            "total": total_loss,
            "self_recon": loss_self,
            "cross_recon": loss_cross,
            "sparsity": loss_sparsity,
            "recon_u": loss_recon_u,
            "recon_c": loss_recon_c,
        }
    
    def sync_forced_shared_post_step(self):
        with torch.no_grad():
            self.decoder_c.weight.data[:, self.forced_shared_indices] = \
                self.decoder_u.weight.data[:, self.forced_shared_indices].clone()

    def compute_fve(self, x_u: torch.Tensor, x_c: torch.Tensor, outputs: dict) -> dict:
        x_u_hat = outputs["x_u_hat"]
        x_c_hat = outputs["x_c_hat"]
        
        var_u = x_u.var()
        var_c = x_c.var()
        
        mse_u = F.mse_loss(x_u, x_u_hat)
        mse_c = F.mse_loss(x_c, x_c_hat)
        
        fve_u = 1.0 - mse_u / (var_u + 1e-8)
        fve_c = 1.0 - mse_c / (var_c + 1e-8)
        
        return {"fve_u": fve_u.item(), "fve_c": fve_c.item()}
    
    def compute_dead_neurons(self, z_u: torch.Tensor, z_c: torch.Tensor) -> float:
        z_combined = z_u + z_c
        active_per_feature = (z_combined.abs() > 0).float().sum(dim=0)
        dead_fraction = (active_per_feature == 0).float().mean().item()
        return dead_fraction
    
    def compute_l0_sparsity(self, z_u: torch.Tensor, z_c: torch.Tensor) -> dict:
        l0_u = (z_u.abs() > 0).float().sum(dim=-1).mean().item()
        l0_c = (z_c.abs() > 0).float().sum(dim=-1).mean().item()
        return {"l0_u": l0_u, "l0_c": l0_c}
    
    def get_decoder_weights(self) -> dict:
        return {
            "W_u_dec": self.decoder_u.weight.data.clone(),
            "W_c_dec": self.decoder_c.weight.data.clone(),
        }
    
    def get_encoder_weights(self) -> dict:
        return {
            "W_u_enc": self.encoder_u.weight.data.clone(),
            "W_c_enc": self.encoder_c.weight.data.clone(),
            "b_u_enc": self.encoder_u.bias.data.clone(),
            "b_c_enc": self.encoder_c.bias.data.clone(),
        }


def create_crosscoder(
    model_name: str,
    component: str,
    token_type: str,
    expansion_factor: Optional[int] = None,
    topk: Optional[int] = None,
) -> SPARCCrossCoder:
    from .utils import get_activation_dim, get_expansion_factor, get_topk_for_token_type
    
    input_dim = get_activation_dim(model_name, component)
    if expansion_factor is None:
        expansion_factor = get_expansion_factor(component)
    if topk is None:
        topk = get_topk_for_token_type(token_type, component)
    
    return SPARCCrossCoder(
        input_dim=input_dim,
        expansion_factor=expansion_factor,
        topk=topk,
        forced_shared_fraction=config.FORCED_SHARED_FRACTION,
        seed=config.SEED,
    )

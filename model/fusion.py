from __future__ import annotations

import torch
from torch import nn


class ResidualZoneFusion(nn.Module):
    """Residual multiplicative gate used to couple zone and LM branches."""

    def __init__(self, hidden_size: int, alpha_init: float = 0.1) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def modulation(self, zone_hidden: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.clamp(0.0, 1.0)
        return 1.0 + alpha * torch.tanh(self.projection(zone_hidden))

    def forward(
        self,
        generation_hidden: torch.Tensor,
        zone_hidden: torch.Tensor,
        *,
        enabled: bool,
        stop_gradient: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if enabled:
            modulation = self.modulation(zone_hidden)
        else:
            modulation = torch.ones_like(generation_hidden)
        applied_modulation = modulation.detach() if stop_gradient else modulation
        return generation_hidden * applied_modulation, modulation

    def set_trainable(self, trainable: bool) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = trainable


def clamp_fusion_alpha(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, ResidualZoneFusion):
            with torch.no_grad():
                child.alpha.clamp_(0.0, 1.0)


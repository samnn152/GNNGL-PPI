import torch
import torch.nn as nn


class ScalarGlobalLocalFusionNode(nn.Module):
    def __init__(self, initial_alpha=0.5, preserve_sum_scale=True):
        super().__init__()
        initial_alpha = min(max(initial_alpha, 1e-6), 1 - 1e-6)
        initial_logit = torch.logit(torch.tensor(initial_alpha, dtype=torch.float32))
        self.alpha_logit = nn.Parameter(initial_logit)
        self.preserve_sum_scale = preserve_sum_scale

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_logit)

    def forward(self, global_x, local_x):
        alpha = self.alpha
        fused = alpha * global_x + (1 - alpha) * local_x
        if self.preserve_sum_scale:
            return 2 * fused
        return fused

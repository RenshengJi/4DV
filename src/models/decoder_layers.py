"""
Decoder layers migrated from storm.
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def modulate(x, shift=None, scale=None):
    """Modulate input tensor with shift and scale parameters."""
    if shift is None and scale is None:
        return x
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class ModulatedLinearLayer(nn.Module):
    """Modulated linear layer with adaptive layer normalization."""

    def __init__(self, in_channels, hidden_channels=64, condition_channels=768, out_channels=3):
        super().__init__()
        self.linear = nn.Linear(in_channels, hidden_channels)
        self.norm = nn.LayerNorm(hidden_channels, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_channels, 2 * hidden_channels, bias=True)
        )
        self.condition_mapping = nn.Linear(condition_channels, hidden_channels)
        self.output = nn.Linear(hidden_channels, out_channels)
        self.gradient_checkpointing = False

    def gradient_checkpointing_enable(self, enable=True):
        """Enable or disable gradient checkpointing."""
        self.gradient_checkpointing = enable

    def _forward_impl(self, x, c):
        """Implementation of forward pass (for gradient checkpointing)."""
        x = self.linear(x)
        c = self.condition_mapping(c.squeeze(1))
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x_shape = x.shape
        x = modulate(self.norm(x.reshape(x_shape[0], -1, x.shape[-1])), shift, scale)
        x = self.output(x)
        x = x.reshape(*x_shape[:-1], -1)
        return x

    def forward(self, x, c, chunk_size=None):
        """
        Forward pass with optional chunking.

        Args:
            x: Input tensor [N, ..., in_channels]
            c: Condition tensor [1, embed_dim] or [N, embed_dim]
            chunk_size: If specified, process N views in chunks to save memory

        Returns:
            Output tensor [N, ..., out_channels]
        """
        if chunk_size is None or x.shape[0] <= chunk_size:
            if self.gradient_checkpointing and self.training:
                return checkpoint(self._forward_impl, x, c, use_reentrant=False)
            else:
                return self._forward_impl(x, c)

        # Process in chunks
        results = []
        N = x.shape[0]
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk_x = x[start:end]
            if self.gradient_checkpointing and self.training:
                chunk_out = checkpoint(self._forward_impl, chunk_x, c, use_reentrant=False)
            else:
                chunk_out = self._forward_impl(chunk_x, c)
            results.append(chunk_out)
        return torch.cat(results, dim=0)

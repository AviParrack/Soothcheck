from dataclasses import dataclass
import torch
import torch.nn as nn
from typing import Optional
from .linear_probe import BaseProbe, ProbeConfig


@dataclass
class MultiLinearProbeConfig(ProbeConfig):
    """Configuration for multi-output linear probe."""

    output_size: int
    normalize_weights: bool = True
    bias: bool = False


class MultiLinearProbe(BaseProbe[MultiLinearProbeConfig]):
    """Linear probe that learns multiple directions simultaneously.

    This probe is particularly useful for:
    1. Multi-class classification (one-vs-rest)
    2. Learning multiple related features simultaneously
    3. Cases where feature directions might be interdependent
    """

    def __init__(self, config: MultiLinearProbeConfig):
        super().__init__(config)

        # Initialize linear layer
        self.linear = nn.Linear(config.input_size, config.output_size, bias=config.bias)

        # Initialize weights using Kaiming initialization
        nn.init.kaiming_uniform_(self.linear.weight, nonlinearity="linear")
        if config.bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the probe.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Tensor of shape (batch_size, output_size)
        """
        return self.linear(x)

    def get_directions(self) -> torch.Tensor:
        """Get all learned probe directions.

        Returns:
            Tensor of shape (output_size, input_size)
        """
        directions = self.linear.weight.data

        if self.config.normalize_weights:
            # Normalize each direction independently
            norms = torch.norm(directions, dim=1, keepdim=True)
            directions = directions / (norms + 1e-8)

        return directions

    def get_direction(self, index: Optional[int] = None) -> torch.Tensor:
        """Get a specific learned probe direction.

        Args:
            index: Which direction to return (default: 0)

        Returns:
            Tensor of shape (input_size,)
        """
        index = index if index is not None else 0
        if index >= self.config.output_size:
            raise ValueError(f"Invalid direction index: {index}")

        directions = self.get_directions()
        return directions[index]

    def get_biases(self) -> torch.Tensor:
        """Get all learned biases if they exist."""
        return self.linear.bias.data if self.config.bias else None

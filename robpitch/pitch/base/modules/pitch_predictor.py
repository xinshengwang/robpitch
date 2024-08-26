# Copyright (c) 2024 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn

from robpitch.base.blocks.common import LayerNorm, Conv1d, ProjectionLayer


class ConvPitchPredictor(nn.Module):
    def __init__(
        self,
        num_pitch_classes: int = 1200,
        num_energy_classes: int = 400,
        input_dim: int = 80,
        hidden_dim: int = 512,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout_rate: float = 0.1,
        include_energy: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.include_energy = include_energy
        self.conv_layers = nn.ModuleList()
        self.residual_layers = nn.ModuleList()

        # Initial convolution layer
        self.initial_conv = Conv1d(input_dim, hidden_dim, kernel_size)

        # Build convolutional and residual layers
        for _ in range(num_layers):
            self.conv_layers.append(
                nn.Sequential(
                    LayerNorm(hidden_dim),
                    Conv1d(hidden_dim, hidden_dim, kernel_size),
                    nn.ELU(),
                    nn.Dropout(dropout_rate),
                )
            )
            self.residual_layers.append(Conv1d(hidden_dim, hidden_dim, kernel_size))

        # Normalization layer for latent representation
        self.latent_norm = LayerNorm(hidden_dim)

        # Projection layer for pitch predictions
        self.pitch_projection = ProjectionLayer(
            hidden_dim,
            num_pitch_classes,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
        )

        # Optional projection layer for energy predictions
        if include_energy:
            self.energy_projection = ProjectionLayer(
                hidden_dim,
                num_energy_classes,
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """pitch predictor forward

        Args:
            x (torch.Tenor): with shape [B, D, T], where T is the sequence length.
        """
        residual = 0
        # Apply convolutional and residual layers
        x = self.initial_conv(x)
        for conv_layer, res_layer in zip(self.conv_layers, self.residual_layers):
            conv_output = conv_layer(x)
            residual += res_layer(conv_output)

        # Normalize and project to pitch and optionally energy classes
        x = residual / self.num_layers
        latent = self.latent_norm(x)
        pitch_logits = self.pitch_projection(latent)
        energy_logits = (
            None if not self.include_energy else self.energy_projection(latent)
        )

        return pitch_logits, energy_logits, latent


# test
if __name__ == "__main__":

    model = ConvPitchPredictor(include_energy=True)
    input_tensor = torch.randn(10, 80, 300)
    pitch, energy, latent = model(input_tensor)

    print("pitch shape: ", pitch.shape)
    print("energy shape: ", energy.shape)
    print("latent shape: ", latent.shape)
    print("+--------------------+\n|   Test Successful  |\n+--------------------+")

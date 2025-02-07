from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.utils import BaseOutput, logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class SMPLEmbedderOutput(BaseOutput):
    """
    The output of [`ExtraCondEmbedderModel`].
    """

    latent_states: torch.Tensor


class SMPLEmbedderModel(ModelMixin, ConfigMixin):
    """
    The ExtraCondEmbedder Model
    """

    @register_to_config
    def __init__(
        self,
        in_channels: int,
        out_shape: Tuple[int, int],
    ):
        super().__init__()

        self.out_shape = out_shape

        self.linear = nn.Linear(in_channels, out_shape[0] * out_shape[1])

    def forward(self, x: torch.Tensor) -> SMPLEmbedderOutput:

        latent_states = self.linear(x)
        latent_states = latent_states.view(-1, *self.out_shape)

        return SMPLEmbedderOutput(latent_states=latent_states)

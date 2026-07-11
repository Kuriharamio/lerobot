#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration for the FASTER Pi0 policy.

FASTER (Fast Action Sampling for ImmediaTE Reaction) augments the standard
Pi0 flow-matching policy with a Horizon-Aware Schedule (HAS) that assigns
each action-horizon index its own noise level during training and inference,
enabling early stopping of the denoising loop once the immediately needed
actions are clean.

Reference: "FASTER: Fast Action Sampling for ImmediaTE Reaction" (2025).
"""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi0.configuration_pi0 import PI0Config


@PreTrainedConfig.register_subclass("faster_pi0")
@dataclass
class FasterPI0Config(PI0Config):
    """Configuration for FasterPI0Policy.

    Inherits all fields from PI0Config and adds the four HAS hyperparameters
    that control the Horizon-Aware Schedule used during training and inference.

    New fields
    ----------
    has_alpha : float
        Exponent controlling how quickly the hit-time decays along the horizon.
        A larger value causes later actions to start denoising sooner relative
        to earlier ones.  Default: 0.6.
    has_u_d : float
        Hit-time for the first *valid* (non-prefix) action step, i.e. the noise
        level at which that step begins to be denoised.  Must be in (0, 1).
        Typically set to ``(N-1)/N`` where ``N`` is the number of inference
        steps.  Default: 0.9.
    has_mix_prob : float
        Probability ``p`` of sampling a HAS trajectory (vs. the standard
        constant-time trajectory) for each training example.  Default: 0.5.
    has_d_max : int
        Maximum simulated inference-delay length used during training.  The
        prefix length ``d`` is sampled uniformly from ``[0, has_d_max]``.
        Must satisfy ``0 <= has_d_max < chunk_size``.  Default: 10.
    """

    # ------------------------------------------------------------------ #
    # Horizon-Aware Schedule (HAS) hyperparameters                        #
    # ------------------------------------------------------------------ #
    has_alpha: float = 0.6
    has_u_d: float = 0.9
    has_mix_prob: float = 0.5
    has_d_max: int = 10

    def __post_init__(self):
        # Run all PI0Config (and PreTrainedConfig) validation first.
        super().__post_init__()

        if not (0.0 < self.has_u_d < 1.0):
            raise ValueError(
                f"`has_u_d` must be in the open interval (0, 1), got {self.has_u_d}."
            )
        if not (0.0 <= self.has_mix_prob <= 1.0):
            raise ValueError(
                f"`has_mix_prob` must be in [0, 1], got {self.has_mix_prob}."
            )
        if self.has_d_max < 0:
            raise ValueError(
                f"`has_d_max` must be >= 0, got {self.has_d_max}."
            )
        if self.has_d_max >= self.chunk_size:
            raise ValueError(
                f"`has_d_max` ({self.has_d_max}) must be strictly less than "
                f"`chunk_size` ({self.chunk_size})."
            )
        if self.has_alpha <= 0.0:
            raise ValueError(
                f"`has_alpha` must be positive, got {self.has_alpha}."
            )

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/mlp.py


from typing import Callable, Optional

from torch import Tensor, nn


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# Optional Transformer Engine for FP8
try:
    import transformer_engine.pytorch as te  # type: ignore
    from transformer_engine.common import recipe as te_recipe  # type: ignore
    TE_AVAILABLE = True
except Exception:
    TE_AVAILABLE = False


class MlpFP8(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
        fp8_recipe=None,
    ) -> None:
        super().__init__()
        if not TE_AVAILABLE:
            raise ImportError("Transformer Engine is required for MlpFP8. Please install 'transformer_engine'.")

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # Use TE Linear layers to enable FP8 when autocast is active
        self.fc1 = te.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = te.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

        # Default FP8 recipe (same default as attention FP8)
        self.fp8_recipe = fp8_recipe if fp8_recipe is not None else (
            te_recipe.DelayedScaling(margin=0, fp8_format=te_recipe.Format.E4M3) if TE_AVAILABLE else None
        )

    def forward(self, x: Tensor) -> Tensor:
        if not TE_AVAILABLE:
            raise RuntimeError("Transformer Engine is not available for FP8 MLP forward path.")

        # Enable FP8 autocast for TE linear ops
        with te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe):
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
        return x

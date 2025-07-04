# Copyright (c) 2025 Minmin Gong
#

# Based on https://github.com/microsoft/TRELLIS/blob/main/trellis/models/sparse_structure_flow.py

from typing import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

from ..Modules.Spatial import Patchify, Unpatchify
from ..Modules.Transformer import AbsolutePositionEmbedder, ModulatedTransformerCrossBlock
from ..Modules.Utils import ConvertModuleToFp16

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size = 256, device: Optional[torch.device] = None):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias = True, device = device),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias = True, device = device),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def TimestepEmbedding(t, dim, max_period = 10000):
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: a 1-D Tensor of N indices, one per batch element.
                These may be fractional.
            dim: the dimension of the output.
            max_period: controls the minimum frequency of the embeddings.

        Returns:
            an (N, D) Tensor of positional embeddings.
        """

        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(start = 0, end = half, dtype = torch.float32) / half
        ).to(device = t.device)
        args = t.unsqueeze(-1).float() * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim = -1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, : 1])], dim = -1)
        return embedding

    def forward(self, t):
        t_freq = self.TimestepEmbedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class SparseStructureFlowModel(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 2,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.resolution = resolution
        self.in_channels = in_channels
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.t_embedder = TimestepEmbedder(model_channels, device = device)
        if share_mod:
            self.ada_ln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias = True, device = device)
            )

        if pe_mode == "ape":
            pos_embedder = AbsolutePositionEmbedder(model_channels, 3)
            coords = torch.meshgrid(*[torch.arange(res, device = device) for res in [resolution // patch_size] * 3], indexing = "ij")
            coords = torch.stack(coords, dim = -1).reshape(-1, 3)
            pos_emb = pos_embedder(coords)
            self.register_buffer("pos_emb", pos_emb)

        self.input_layer = nn.Linear(in_channels * (patch_size ** 3), model_channels, device = device)

        self.blocks = nn.ModuleList([
            ModulatedTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads = self.num_heads,
                mlp_ratio = self.mlp_ratio,
                attn_mode = "full",
                use_rope = (pe_mode == "rope"),
                share_mod = share_mod,
                qk_rms_norm = self.qk_rms_norm,
                qk_rms_norm_cross = self.qk_rms_norm_cross,
                device = device,
            )
            for _ in range(num_blocks)
        ])

        self.out_layer = nn.Linear(model_channels, out_channels * (patch_size ** 3), device = device)

        if device != "meta":
            self.InitializeWeights()
        if use_fp16:
            self.ConvertToFp16()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """

        return next(self.parameters()).device

    def ConvertToFp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """

        self.blocks.apply(ConvertModuleToFp16)

    def InitializeWeights(self) -> None:
        # Initialize transformer layers:
        def BasicInit(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(BasicInit)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std = 0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std = 0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        if self.share_mod:
            nn.init.constant_(self.ada_ln_modulation[-1].weight, 0)
            nn.init.constant_(self.ada_ln_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.ada_ln_modulation[-1].weight, 0)
                nn.init.constant_(block.ada_ln_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        assert [*x.shape] == [x.shape[0], self.in_channels, *[self.resolution] * 3], \
                f"Input shape mismatch, got {x.shape}, expected {[x.shape[0], self.in_channels, *[self.resolution] * 3]}"

        h = Patchify(x, self.patch_size)
        h = h.view(*h.shape[: 2], -1).permute(0, 2, 1).contiguous()

        h = self.input_layer(h)
        h = h + self.pos_emb.unsqueeze(0)
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.ada_ln_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)
        h = h.type(self.dtype)
        cond = cond.type(self.dtype)
        for block in self.blocks:
            h = block(h, t_emb, cond)
        h = h.type(x.dtype)
        h = functional.layer_norm(h, h.shape[-1 :])
        h = self.out_layer(h)

        h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.resolution // self.patch_size] * 3)
        h = Unpatchify(h, self.patch_size).contiguous()

        return h

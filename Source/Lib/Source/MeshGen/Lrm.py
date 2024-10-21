# Copyright (c) 2024 Minmin Gong
#

# Take InstantMesh/src/models/lrm_mesh.py as a reference
# The major difference is
#   1. Removing the batch_size, since it's always 1.
#   2. Tons of code clean and simplification is taken placed.
#   3. Replace FlexiCubes with marching cubes.

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.models.decoder.transformer import TriplaneTransformer
from src.models.encoder.dino_wrapper import DinoWrapper

class Lrm(nn.Module):
    def __init__(self,
                 encoder_feat_dim : int = 768,
                 transformer_dim : int = 1024,
                 transformer_layers : int = 16,
                 transformer_heads : int = 16,
                 triplane_low_res : int = 32,
                 triplane_high_res : int = 64,
                 triplane_dim : int = 80):
        super(Lrm, self).__init__()

        this_py_dir = Path(__file__).parent.resolve()
        pretrained_dir = this_py_dir.joinpath("Models/dino-vitb16")
        reload_from_network = True
        if pretrained_dir.exists():
            print(f"Load from {pretrained_dir}.")
            try:
                self.encoder = DinoWrapper(
                    model_name = pretrained_dir,
                    freeze = False,
                )
                reload_from_network = False
            except:
                print(f"Failed. Retry from network.")

        if reload_from_network:
            self.encoder = DinoWrapper(
                model_name = "facebook/dino-vitb16",
                freeze = False,
            )

            pretrained_dir.mkdir(parents = True, exist_ok = True)
            self.encoder.model.save_pretrained(pretrained_dir)
            self.encoder.processor.save_pretrained(pretrained_dir)

        self.transformer = TriplaneTransformer(
            inner_dim = transformer_dim,
            num_layers = transformer_layers,
            num_heads = transformer_heads,
            image_feat_dim = encoder_feat_dim,
            triplane_low_res = triplane_low_res,
            triplane_high_res = triplane_high_res,
            triplane_dim = triplane_dim,
        )

    def GenNeRF(self, images, cameras):
        image_feats = self.encoder(images.unsqueeze(0), cameras.unsqueeze(0))
        image_feats = image_feats.reshape(1, image_feats.shape[0] * image_feats.shape[1], image_feats.shape[2])

        planes = self.transformer(image_feats)
        assert(planes.shape[0] == 1)
        return planes.squeeze(0)

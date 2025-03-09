# Copyright (c) 2024 Minmin Gong
#

from pathlib import Path

import onnx
import onnx2torch
import torch

if __name__ == "__main__":
    this_py_dir = Path(__file__).parent.resolve()

    u2net_model_path = this_py_dir / "RembgModels/u2net.onnx"
    onnx_model = onnx.load(u2net_model_path)
    u2net = onnx2torch.convert(onnx_model)
    torch.save(u2net, this_py_dir / "RembgModels/u2net.pt")

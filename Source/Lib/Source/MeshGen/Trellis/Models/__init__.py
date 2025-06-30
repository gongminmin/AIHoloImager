# Copyright (c) 2025 Minmin Gong
#

# Based on https://github.com/microsoft/TRELLIS/blob/main/trellis/models/__init__.py

from torch.nn.utils import skip_init

__attributes = {
    "SparseStructureDecoder": "SparseStructureVae",
    "SparseStructureFlowModel": "SparseStructureFlow",
    "SLatMeshDecoder": "StructuredLatentVae",
    "SLatFlowModel": "StructuredLatentFlow",
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        import importlib
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]

def FromPretrained(path: str, **kwargs):
    from pathlib import Path
    import json
    from safetensors.torch import load_file

    config_file_path = Path(f"{path}.json")
    model_file_path = Path(f"{path}.safetensors")

    with open(config_file_path, "r") as file:
        config = json.load(file)
    model = skip_init(__getattr__(config["name"]), **config["args"], **kwargs)
    state_dict = load_file(model_file_path)
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.find("adaLN_modulation") != -1:
            key = key.replace("adaLN_modulation", "ada_ln_modulation")
        new_state_dict[key] = value
    model.load_state_dict(new_state_dict)

    return model

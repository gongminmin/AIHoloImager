# Copyright (c) 2025 Minmin Gong
#

# Based on https://github.com/microsoft/TRELLIS/blob/main/trellis/models/__init__.py

import importlib

import Util

__attributes = {
    "SparseStructureDecoder" : "SparseStructureVae",
    "SparseStructureFlowModel" : "SparseStructureFlow",
    "SLatMeshDecoder" : "StructuredLatentVae",
    "SLatFlowModel" : "StructuredLatentFlow",
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
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

def FromPretrained(path : str, local_dir : str, **kwargs):
    from pathlib import Path
    import json
    from safetensors.torch import load_file

    config_file_path = Path(f"{path}.json")
    model_file_path = Path(f"{path}.safetensors")
    if local_dir != None:
        config_file_path = local_dir / config_file_path
        model_file_path = local_dir / model_file_path

    if not (config_file_path.exists() and model_file_path.exists()):
        path = Path(path)
        repo = f"{path.parts[0]}/{path.parts[1]}"
        model_name = "/".join(path.parts[2 :])

        config_file_name = f"{model_name}.json"
        config_file_path = local_dir / config_file_name
        if not config_file_path.exists():
            print(f"Downloading \"{config_file_name}\" to \"{local_dir}\"")
            Util.DownloadFile(Util.GenHuggingFaceLink(repo, config_file_name), config_file_path)

        model_file_name = f"{model_name}.safetensors"
        model_file_path = local_dir / model_file_name
        if not model_file_path.exists():
            print(f"Downloading \"{model_file_name}\" to \"{local_dir}\"")
            Util.DownloadFile(Util.GenHuggingFaceLink(repo, model_file_name), model_file_path)

    with open(config_file_path, "r") as file:
        config = json.load(file)
    model = __getattr__(config["name"])(**config["args"], **kwargs)
    model.load_state_dict(load_file(model_file_path))

    return model

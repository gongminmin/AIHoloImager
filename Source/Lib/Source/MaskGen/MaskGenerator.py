# Copyright (c) 2024 Minmin Gong
#

from pathlib import Path
from rembg import new_session, remove
import shutil

class MaskGenerator:
    def __init__(self):
        this_py_dir = Path(__file__).parent.resolve()
        u2net_model_path = this_py_dir.joinpath("Models/u2net.onnx")
        if u2net_model_path.exists():
            self.session = new_session("u2net_custom", model_path = u2net_model_path)
        else:
            print("Downloading pre-trained mask generator models...")
            self.session = new_session("u2net")
            shutil.copyfile(self.session.download_models(), u2net_model_path)

    def Gen(self, input_img):
        return remove(input_img, session = self.session, only_mask = True, post_process_mask = True)

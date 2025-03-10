# Copyright (c) 2025 Minmin Gong
#

def SeedRandom(seed : int):
    import random
    random.seed(seed)

    import numpy
    numpy.random.seed(seed)

    import torch
    torch.manual_seed(seed)

def InitPySys():
    import os
    os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"

    SeedRandom(42)

    if os.name == "nt":
        from pathlib import Path
        this_py_dir = Path(__file__).parent.resolve()

        # pywintypes imports modules in these directories
        import sys
        sys.path.append(str(this_py_dir / "Python/Lib/site-packages/win32"))
        sys.path.append(str(this_py_dir / "Python/Lib/site-packages/win32/lib"))

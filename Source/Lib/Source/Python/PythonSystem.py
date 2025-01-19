def InitPySys():
    import os
    os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"

    import Util
    Util.SeedRandom(42)

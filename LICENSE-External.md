# Third-Party Licenses

This project uses the following open-source libraries and tools. We are grateful to all the maintainers and contributors of these projects.

## Direct Dependencies

### assimp
- **Repository**: https://github.com/assimp/assimp
- **License**: [BSD 3-Clause License](https://github.com/assimp/assimp/blob/master/LICENSE)
- **Copyright**: assimp team

### cxxopts
- **Repository**: https://github.com/jarro2783/cxxopts
- **License**: [MIT License](https://github.com/jarro2783/cxxopts/blob/master/LICENSE)
- **Copyright**: Jarryd Beck

### diff-gaussian-rasterization
- **Repository**: https://github.com/graphdeco-inria/diff-gaussian-rasterization
- **License**: [Gaussian-Splatting License](https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/LICENSE.md) (Non-commercial only)
- **Copyright**: INRIA

**Note**, we reimplement a compute shader based inference process in our code base, by taking the original code as a reference.

### dinov2
- **Repository**: https://github.com/facebookresearch/dinov2
- **Code & Pretrained Model License**: [Apache License 2.0](https://github.com/facebookresearch/dinov2/blob/main/LICENSE)
- **Copyright**: Meta Platforms, Inc. and affiliates

### DirectX-Headers
- **Repository**: https://github.com/microsoft/DirectX-Headers
- **License**: [MIT License](https://github.com/microsoft/DirectX-Headers/blob/main/LICENSE)
- **Copyright**: Microsoft Corporation

### DirectXShaderCompiler
- **Repository**: https://github.com/microsoft/DirectXShaderCompiler
- **License**: [LLVM Release License](https://github.com/microsoft/DirectXShaderCompiler/blob/main/LICENSE.TXT)
- **Copyright**: Microsoft Corporation

**Note**, we don't reference its code, only call its binaries during development.

### glm
- **Repository**: https://github.com/g-truc/glm
- **License**: [MIT License](https://github.com/g-truc/glm/blob/master/copying.txt)
- **Copyright**: G-Truc Creation

### Intrinsic
- **Repository**: https://github.com/compphoto/Intrinsic
- **Code & Pretrained Model License**: [Academic use only License](https://github.com/compphoto/Intrinsic/blob/main/LICENSE) (Non-commercial only)
- **Copyright**: Chris Careaga, Yagiz Aksoy, Computational Photography Laboratory

### LightGlue
- **Repository**: https://github.com/cvg/LightGlue
- **Code & Pretrained Model License**: [Apache-2.0 License](https://github.com/cvg/LightGlue/blob/main/LICENSE)
- **Copyright**: Philipp Lindenberger, Paul-Edouard Sarlin, Marc Pollefeys

### MoGe
- **Repository**: https://github.com/microsoft/moge
- **Code License**: [MIT License](https://github.com/microsoft/MoGe/blob/main/LICENSE)
- **Pretrained Model License**: [Apache-2.0 License](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)
- **Copyright**: Microsoft Corporation

**Note**, we implement a simplified inference process in our code base, by taking the original code as a reference.

### numpy
- **Repository**: https://github.com/numpy/numpy
- **License**: [BSD 3-Clause License](https://github.com/numpy/numpy/blob/main/LICENSE.txt)
- **Copyright**: NumPy Developers

**Note**, it's depended via pip package.

### openMVG
- **Repository**: https://github.com/openMVG/openMVG
- **License**: [MPL-2.0 License](https://github.com/openMVG/openMVG/blob/develop/LICENSE)
- **Copyright**: [OpenMVG authors](https://github.com/openMVG/openMVG/blob/develop/AUTHORS)

### Python
- **Repository**: https://github.com/python/cpython
- **License**: [PSF License Agreement](https://docs.python.org/3/license.html#psf-license)
- **Copyright**: Python Software Foundation

### PyTorch
- **Repository**: https://github.com/pytorch/pytorch
- **License**: [BSD 3-Clause License](https://github.com/pytorch/pytorch/blob/main/LICENSE)
- **Copyright**: Facebook, Inc.

**Note**, it's depended via pip package.

### safetensors
- **Repository**: https://github.com/safetensors/safetensors
- **License**: [Apache-2.0 License](https://github.com/safetensors/safetensors/blob/main/LICENSE)

**Note**, it's depended via pip package.

### SPIRV-Reflect
- **Repository**: https://github.com/KhronosGroup/SPIRV-Reflect
- **License**: [Apache-2.0 License](https://github.com/KhronosGroup/SPIRV-Reflect/blob/main/LICENSE)
- **Copyright**: Google Inc.

### stb
- **Repository**: https://github.com/nothings/stb
- **License**: [Public Domain](https://github.com/nothings/stb/blob/master/LICENSE)

### SuperPoint
- **Repository**: https://github.com/magicleap/SuperPointPretrainedNetwork
- **Code License**: [Apache-2.0 License](https://github.com/cvg/LightGlue/blob/main/LICENSE)
- **Pretrained Model License**: [Academic use only License](https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/LICENSE) (Non-commercial only)
- **Copyright**: Magic Leap, Inc.

**Note**, we take the inference code from LightGlue, pretrained model from the original SuperPoint.

### torchvision
- **Repository**: https://github.com/pytorch/vision
- **License**: [BSD 3-Clause License](https://github.com/pytorch/vision/blob/main/LICENSE)
- **Copyright**: Soumith Chintala

**Note**, it's depended via pip package.

### tqdm
- **Repository**: https://github.com/tqdm/tqdm
- **License**: [MPL-2.0 License](https://github.com/tqdm/tqdm/blob/master/LICENCE)
- **Copyright**: Casper da Costa-Luis

**Note**, it's depended via pip package.

### TRELLIS
- **Repository**: https://github.com/Microsoft/TRELLIS
- **Code & Pretrained Model License**: [MIT License](https://github.com/microsoft/TRELLIS/blob/main/LICENSE)
- **Copyright**: Microsoft Corporation

**Note**, we implement a simplified inference process in our code base, by taking the original code as a reference.

### U-2-Net
- **Repository**: https://github.com/xuebinqin/U-2-Net
- **Code & Pretrained Model License**: [Apache-2.0 License](https://github.com/xuebinqin/U-2-Net/blob/master/LICENSE)

### volk
- **Repository**: https://github.com/zeux/volk
- **License**: [MIT License](https://github.com/zeux/volk/blob/master/LICENSE.md)
- **Copyright**: Arseny Kapoulkine

### Vulkan-Headers
- **Repository**: https://github.com/KhronosGroup/Vulkan-Headers
- **License**: [Apache-2.0 License](https://github.com/KhronosGroup/Vulkan-Headers/blob/main/LICENSE.md)
- **Copyright**: The Khronos Group Inc.

### Vulkan radix sort
- **Repository**: https://github.com/jaesung-cs/vulkan_radix_sort
- **License**: [MIT License](https://github.com/jaesung-cs/vulkan_radix_sort/blob/master/LICENSE)
- **Copyright**: jaesung-cs

**Note**, we reimplement a compute shader based sort in our code base, by taking the original code as a reference.

### xatlas
- **Repository**: https://github.com/jpcy/xatlas
- **License**: [MIT License](https://github.com/jpcy/xatlas/blob/master/LICENSE)
- **Copyright**: Jonathan Young

### zlib
- **Repository**: https://github.com/madler/zlib
- **License**: [zlib License](http://zlib.net/zlib_license.html)
- **Copyright**: Jean-loup Gailly and Mark Adler

## Transitive Dependencies

### gdown
- **Repository**: https://github.com/wkentaro/gdown
- **License**: [MIT License](https://github.com/wkentaro/gdown/blob/main/LICENSE)
- **Copyright**: Kentaro Wada

**Note**, it's depended via pip package. We only call it's binary during development.

### easyexif
- **Repository**: https://github.com/mayanklahiri/easyexif
- **License**: [BSD License](https://github.com/mayanklahiri/easyexif/blob/master/LICENSE)
- **Copyright**: Mayank Lahiri

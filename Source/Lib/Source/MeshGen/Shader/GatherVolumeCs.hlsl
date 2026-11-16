// Copyright (c) 2025-2026 Minmin Gong
//

#include "ColorSpace.hlslh"

static const uint32_t BlockDim = 16;

cbuffer param_cb
{
    uint32_t grid_res;
    uint32_t size;
};

Texture3D<uint32_t> index_volume;
#ifdef __spirv__
[[vk::image_format("r16f")]]
#endif
Buffer<float> density_features;
#ifdef __spirv__
[[vk::image_format("r16f")]]
#endif
Buffer<float> deformation_features;

#ifdef __spirv__
[[vk::image_format("rgba16f")]]
#endif
RWTexture3D<float4> density_deformation_volume;

[numthreads(BlockDim, BlockDim, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (any(dtid >= size))
    {
        return;
    }

    const int32_t3 CubeOffsets[] = {
        int32_t3(-1, -1, -1),
        int32_t3(0, -1, -1),
        int32_t3(-1, 0, -1),
        int32_t3(0, 0, -1),
        int32_t3(-1, -1, 0),
        int32_t3(0, -1, 0),
        int32_t3(-1, 0, 0),
        int32_t3(0, 0, 0),
    };

    float density = 0;
    float3 deformation = 0;
    uint32_t count = 0;
    for (uint32_t i = 0; i < 8; ++i)
    {
        const int32_t3 coord = int32_t3(dtid) + CubeOffsets[i];
        if (all(coord >= 0) && all(coord < grid_res))
        {
            const uint32_t sparse_index = index_volume[coord];
            if (sparse_index != 0)
            {
                const uint32_t index = (sparse_index - 1) * 8 + (7 - i);

                density += density_features[index];
                deformation += float3(deformation_features[index * 3 + 0], deformation_features[index * 3 + 1], deformation_features[index * 3 + 2]);
                count += 1;
            }
        }
    }

    if (count > 0)
    {
        density /= count;
        deformation /= count;

        const float DeformationMultiplier = 4.0f;
        deformation = 1.0f / (grid_res * DeformationMultiplier) * tanh(deformation);
        deformation *= size;
    }
    else
    {
        density = 1;
        deformation = 0;
    }

    density_deformation_volume[dtid.zyx] = float4(density, deformation);
}

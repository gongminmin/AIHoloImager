// Copyright (c) 2025 Minmin Gong
//

static const uint32_t BlockDim = 16;

cbuffer param_cb : register(b0)
{
    uint32_t grid_res;
    uint32_t size;
};

Texture3D<uint32_t> index_volume : register(t0);
Buffer<float> density_features : register(t1);
Buffer<float3> deformation_features : register(t2);
Buffer<float3> color_features : register(t3);

RWTexture3D<float4> density_deformation_volume : register(u0);
RWTexture3D<unorm float4> color_volume : register(u1);

float3 Sigmoid(float3 v)
{
    return 1 / (1 + exp(-v));
}

[numthreads(BlockDim, BlockDim, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (any(dtid.xyz >= size))
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
    float3 color = 0;
    uint32_t count = 0;
    for (uint32_t i = 0; i < 8; ++i)
    {
        const int32_t3 coord = int32_t3(dtid) + CubeOffsets[i];
        if (all(coord >= 0) && all(coord < grid_res))
        {
            const uint32_t sparse_index = index_volume[dtid + CubeOffsets[i]];
            if (sparse_index != 0)
            {
                const uint32_t index = (sparse_index - 1) * 8 + (7 - i);

                density += density_features[index];
                deformation += deformation_features[index];
                color += color_features[index];
                ++count;
            }
        }
    }

    if (count > 0)
    {
        density /= count;
        deformation /= count;
        color /= count;

        const float DeformationMultiplier = 4.0f;
        deformation = 1.0f / (grid_res * DeformationMultiplier) * tanh(deformation);
        deformation *= size;

        color = Sigmoid(color);
    }
    else
    {
        density = 1;
        deformation = 0;
        color = 0.5f;
    }

    density_deformation_volume[dtid.zyx] = float4(density, deformation);
    color_volume[dtid.zyx] = float4(color, 1);
}

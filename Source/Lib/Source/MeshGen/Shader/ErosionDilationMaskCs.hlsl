// Copyright (c) 2025 Minmin Gong
//

static const uint32_t BlockDim = 16;
static const int32_t KernelRadius = 8;

static const uint32_t CacheDim = BlockDim + KernelRadius * 2;

cbuffer param_cb
{
    uint32_t2 texture_size;
    bool erosion;
    uint32_t channel;
};

Texture2D input_tex;

#ifdef __spirv__
[[vk::image_format("r8")]]
#endif
RWTexture2D<unorm float> output_tex;

groupshared bool sh_cache[CacheDim][CacheDim];

[numthreads(BlockDim, BlockDim, 1)]
void main(uint32_t3 group_id : SV_GroupID, uint32_t3 gt_id : SV_GroupThreadID, uint32_t3 dtid : SV_DispatchThreadID, uint32_t group_index : SV_GroupIndex)
{
    for (uint32_t i = group_index; i < CacheDim * CacheDim; i += BlockDim * BlockDim)
    {
        const uint32_t y = i / CacheDim;
        const uint32_t x = i - y * CacheDim;
        uint32_t2 coord = clamp(int32_t2(group_id.xy * BlockDim) - KernelRadius + int32_t2(x, y), 0, texture_size - 1);
        sh_cache[y][x] = input_tex.Load(uint32_t3(coord, 0))[channel] > 0.5f;
        if (erosion)
        {
            sh_cache[y][x] = !sh_cache[y][x];
        }
    }
    GroupMemoryBarrierWithGroupSync();

    [branch]
    if (any(dtid.xy >= texture_size))
    {
        return;
    }

    bool mark = false;
    for (int32_t y = -KernelRadius; y <= KernelRadius; ++y)
    {
        for (int32_t x = -KernelRadius; x <= KernelRadius; ++x)
        {
            if (sh_cache[gt_id.y + KernelRadius + y][gt_id.x + KernelRadius + x])
            {
                mark = true;
            }
        }
    }

    output_tex[dtid.xy] = erosion ? !mark : mark;
}

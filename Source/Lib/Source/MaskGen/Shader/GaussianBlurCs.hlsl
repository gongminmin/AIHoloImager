// Copyright (c) 2024 Minmin Gong
//

static const uint32_t BlockDim = 16;
static const int32_t KernelRadius = 2;

cbuffer param_cb : register(b0)
{
    uint32_t2 texture_size;
    bool x_dir;
    float weights[KernelRadius * 2 + 1];
};

Texture2D<float> input_tex : register(t0);

RWTexture2D<unorm float> output_tex : register(u0);

groupshared float group_cache[BlockDim * (BlockDim + KernelRadius * 2)];

[numthreads(BlockDim, BlockDim, 1)]
void main(uint32_t3 group_id : SV_GroupID, uint32_t3 gt_id : SV_GroupThreadID, uint32_t3 dtid : SV_DispatchThreadID, uint32_t group_index : SV_GroupIndex)
{
    const uint32_t cache_width = BlockDim + (x_dir ? KernelRadius * 2 : 0);
    const uint32_t cache_height = BlockDim + (x_dir ? 0 : KernelRadius * 2);

    const int32_t2 start = int32_t2(x_dir ? KernelRadius : 0, x_dir ? 0 : KernelRadius);
    for (uint32_t i = group_index; i < cache_width * cache_height; i += BlockDim * BlockDim)
    {
        const uint32_t y = i / cache_width;
        const uint32_t x = i - y * cache_width;
        uint32_t2 coord = clamp(int32_t2(group_id.xy * BlockDim) - start + int32_t2(x, y), 0, texture_size - 1);
        group_cache[y * cache_width + x] = input_tex.Load(uint32_t3(coord, 0));
    }
    GroupMemoryBarrierWithGroupSync();

    [branch]
    if (any(dtid.xy >= texture_size))
    {
        return;
    }

    float sum = 0;
    for (int32_t i = -KernelRadius; i <= KernelRadius; ++i)
    {
        const int32_t2 cache_coord = start + gt_id.xy + (x_dir ? int32_t2(i, 0) : int32_t2(0, i));
        sum += group_cache[cache_coord.y * cache_width + cache_coord.x] * weights[i + KernelRadius];
    }

    output_tex[dtid.xy] = x_dir ? sum : (sum > 0.5f);
}

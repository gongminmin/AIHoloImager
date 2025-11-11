// Copyright (c) 2025 Minmin Gong
//

#include "Common.hlslh"

static const uint32_t BlockDim = 16;

cbuffer param_cb
{
    uint32_t2 tex_size;
    uint32_t num_channels;
    uint32_t mip_levels;
    uint32_t4 mip_level_offsets[4];
};

Buffer<float> grad_texture_mips;

RWBuffer<float> grad_texture;

[numthreads(BlockDim, BlockDim, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (any(dtid.xy >= tex_size))
    {
        return;
    }

    // Fig 2 (c)

    float4 accum = 0;
    const uint32_t mip0_texels = tex_size.x * tex_size.y;
    for (uint32_t level = 0; level < mip_levels; ++level)
    {
        const uint32_t2 level_size = MipLevelSize(tex_size, level);
        const float weight = float(level_size.x * level_size.y) / mip0_texels;

        const uint32_t2 coord = dtid.xy >> level;
        const uint32_t offset = MipLevelOffset(mip_level_offsets, level) + (coord.y * level_size.x + coord.x) * num_channels;
        for (uint32_t i = 0; i < num_channels; ++i)
        {
            accum[i] += grad_texture_mips[offset + i] * weight;
        }
    }

    const uint32_t output_offset = (dtid.y * tex_size.x + dtid.x) * num_channels;
    for (uint32_t i = 0; i < num_channels; ++i)
    {
        grad_texture[output_offset + i] = accum[i];
    }
}

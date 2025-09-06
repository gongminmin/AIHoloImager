// Copyright (c) 2025 Minmin Gong
//

static const uint32_t BlockDim = 16;

cbuffer param_cb : register(b0)
{
    uint32_t2 this_level_size;
};

Texture2D last_level_tex : register(t0);

RWTexture2D<float4> this_level_tex : register(u0);

SamplerState mip_sampler : register(s0);

[numthreads(BlockDim, BlockDim, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (any(dtid.xy >= this_level_size))
    {
        return;
    }

    this_level_tex[dtid.xy] = last_level_tex.SampleLevel(mip_sampler, (dtid.xy + 0.5f) / this_level_size, 0);
}

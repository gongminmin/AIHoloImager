// Copyright (c) 2025 Minmin Gong
//

#if ENABLE_MIP
#include "Common.hlslh"
#endif

static const uint32_t BlockDim = 16;

cbuffer param_cb : register(b0)
{
    uint32_t2 gbuffer_size;
    uint32_t2 tex_size;
    uint32_t mip_levels;
};

Texture2D<uint32_t> prim_id_tex : register(t0);
Texture2D texture : register(t1);
Buffer<float2> uv_buff : register(t2);
#if ENABLE_MIP
Buffer<float4> derivative_uv_buff : register(t3);
#endif

RWTexture2D<float4> image : register(u0);

SamplerState tex_sampler : register(s0, space1);

[numthreads(BlockDim, BlockDim, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (any(dtid.xy >= gbuffer_size))
    {
        return;
    }

    uint32_t fi = prim_id_tex[dtid.xy];
    [branch]
    if (fi == 0)
    {
        return;
    }

    const uint32_t index = dtid.y * gbuffer_size.x + dtid.x;
    const float2 uv = uv_buff[index];
#if ENABLE_MIP
    const float level = CalcMipLevel(derivative_uv_buff[index], tex_size, mip_levels);
#else
    const float level = 0;
#endif
    image[dtid.xy] = texture.SampleLevel(tex_sampler, uv, level);
}

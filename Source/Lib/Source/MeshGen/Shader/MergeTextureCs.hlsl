// Copyright (c) 2024-2025 Minmin Gong
//

static const uint32_t BlockDim = 16;

cbuffer param_cb : register(b0)
{
    float4x4 inv_model;

    uint32_t texture_size;
    float inv_scale;
};

Texture3D color_vol_tex : register(t0);
Texture2D pos_tex : register(t1);

SamplerState trilinear_sampler : register(s0);

RWTexture2D<unorm float4> merged_tex : register(u0);

[numthreads(BlockDim, BlockDim, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (any(dtid.xy >= texture_size))
    {
        return;
    }

    [branch]
    if (merged_tex[dtid.xy].a > 0.5f)
    {
        return;
    }

    float4 pos_ws = pos_tex[dtid.xy];
    [branch]
    if (pos_ws.a < 0.5f)
    {
        return;
    }

    float4 pos_os = mul(pos_ws, inv_model);
    pos_os /= pos_os.w;

    const float3 vol_coord = pos_os.zyx * inv_scale + 0.5f;
    merged_tex[dtid.xy] = saturate(color_vol_tex.SampleLevel(trilinear_sampler, vol_coord, 0));
}

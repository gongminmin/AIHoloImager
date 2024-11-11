// Copyright (c) 2024 Minmin Gong
//

static const uint32_t BlockDim = 16;

cbuffer param_cb : register(b0)
{
    uint32_t2 texture_size;
};

Texture2D<float4> input_tex : register(t0);
Texture2D<uint32_t> max_tex : register(t1);

RWTexture2D<float> output_tex : register(u0);

[numthreads(BlockDim, BlockDim, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (any(dtid.xy >= texture_size))
    {
        return;
    }

    float3 color = input_tex.Load(uint3(dtid.xy, 0)).rgb;
    color /= max_tex.Load(uint3(0, 0, 0)) / 1e5f;

    static const float3 Mean = float3(0.485f, 0.456f, 0.406f);
    static const float3 Std = float3(0.229f, 0.224f, 0.225f);
    color = (color - Mean) / Std;

    output_tex[uint32_t2(dtid.x, dtid.y + 0 * texture_size.y)] = color.r;
    output_tex[uint32_t2(dtid.x, dtid.y + 1 * texture_size.y)] = color.g;
    output_tex[uint32_t2(dtid.x, dtid.y + 2 * texture_size.y)] = color.b;
}

// Copyright (c) 2024 Minmin Gong
//

#include "Util.hlslh"

Texture2D photo_tex : register(t0);

SamplerState bilinear_sampler : register(s0);

void main(float3 pos_os : TEXCOORD0,
    float2 photo_tc : TEXCOORD1,
    out float4 color_rt : SV_Target0,
    out float4 pos_rt : SV_Target1)
{
    color_rt = photo_tex.Sample(bilinear_sampler, photo_tc);
    pos_rt = 0;

    if (IsEmpty(color_rt.rgb))
    {
        pos_rt = float4(pos_os, 1);
    }
}
